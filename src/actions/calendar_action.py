"""Calendar draft/confirm action cho lịch cá nhân và hai phòng họp MKAC."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo


CALENDAR_OWNER_EMAIL = os.getenv(
    "CALENDAR_OWNER_EMAIL",
    "son.nguyendinh@meiko.vn",
).strip().lower()
CALENDAR_ROOMS_PATH = Path(
    os.getenv("CALENDAR_ROOMS_PATH", "config/calendar_rooms.json")
)


class CalendarActionError(RuntimeError):
    """Calendar action không thể hoàn thành an toàn."""


@dataclass(frozen=True)
class CalendarRoom:
    id: str
    name: str
    calendar_id: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class CalendarDraft:
    id: str
    session_id: str
    employee_id: str
    title: str
    start: datetime
    end: datetime
    room: CalendarRoom | None
    status: str = "pending"
    event_id: str = ""
    event_url: str = ""
    room_status: str = ""


@dataclass(frozen=True)
class CalendarActionResult:
    kind: str
    answer: str
    draft: CalendarDraft | None = None


ToolRunner = Callable[[str, dict[str, Any]], Awaitable[Any]]
Planner = Callable[[str, datetime], Awaitable[dict[str, Any]]]


class CalendarDraftStore:
    def __init__(self, *, max_items: int = 200, ttl_seconds: float = 15 * 60):
        self._items: "OrderedDict[str, tuple[float, CalendarDraft]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.max_items = max(1, max_items)
        self.ttl_seconds = ttl_seconds

    async def put(self, draft: CalendarDraft) -> None:
        async with self._lock:
            self._items[draft.session_id] = (
                time.monotonic() + self.ttl_seconds,
                draft,
            )
            self._items.move_to_end(draft.session_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    async def get(self, session_id: str) -> CalendarDraft | None:
        async with self._lock:
            cached = self._items.get(session_id)
            if cached is None:
                return None
            expiry_at, draft = cached
            if time.monotonic() > expiry_at:
                self._items.pop(session_id, None)
                return None
            self._items.move_to_end(session_id)
            return draft

    async def discard(self, session_id: str) -> CalendarDraft | None:
        async with self._lock:
            cached = self._items.pop(session_id, None)
            return cached[1] if cached else None


class CalendarActionService:
    def __init__(
        self,
        *,
        tool_runner: ToolRunner,
        planner: Planner,
        rooms_path: Path = CALENDAR_ROOMS_PATH,
        owner_email: str = CALENDAR_OWNER_EMAIL,
        store: CalendarDraftStore | None = None,
    ):
        self.tool_runner = tool_runner
        self.planner = planner
        self.owner_email = owner_email.strip().lower()
        self.store = store or CalendarDraftStore()
        config = json.loads(rooms_path.read_text(encoding="utf-8"))
        self.timezone = ZoneInfo(config.get("timezone", "Asia/Ho_Chi_Minh"))
        self.rooms = tuple(
            CalendarRoom(
                id=item["id"],
                name=item["name"],
                calendar_id=item["calendar_id"],
                aliases=tuple(item.get("aliases", [])),
            )
            for item in config.get("rooms", [])
        )
        self._confirm_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize(
            "NFD",
            (text or "").lower().replace("đ", "d"),
        )
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", normalized).strip()

    @classmethod
    def is_create_request(cls, question: str) -> bool:
        text = cls._normalize(question)
        has_create = any(
            marker in text
            for marker in (
                "tao lich",
                "tao su kien",
                "dat lich",
                "dat phong",
                "them su kien",
                "schedule event",
                "create event",
                "book room",
            )
        ) or any(marker in (question or "") for marker in ("予定を作成", "会議室を予約"))
        has_calendar_context = any(
            marker in text
            for marker in ("lich", "su kien", "phong", "hop", "event", "room")
        )
        return has_create and has_calendar_context

    @classmethod
    def is_confirm_request(cls, question: str) -> bool:
        text = cls._normalize(question)
        return text in {
            "xac nhan",
            "xac nhan tao lich",
            "xac nhan dat lich",
            "dong y tao lich",
            "dong y dat lich",
            "confirm",
            "confirm event",
        } or any(marker in (question or "") for marker in ("予定を確定", "作成を確認"))

    @classmethod
    def is_cancel_request(cls, question: str) -> bool:
        text = cls._normalize(question)
        return text in {
            "huy",
            "huy ban nhap",
            "huy dat lich",
            "cancel",
            "cancel draft",
        } or "下書きをキャンセル" in (question or "")

    @classmethod
    def is_room_availability_request(cls, question: str) -> bool:
        text = cls._normalize(question)
        has_room = bool(
            re.search(r"\b(phong hop|phong|meeting room|room)\s*(?:so\s*)?[13]\b", text)
        )
        has_check = any(
            marker in text
            for marker in (
                "kiem tra",
                "co trong",
                "con trong",
                "ranh",
                "ban",
                "availability",
                "available",
                "free",
                "busy",
            )
        )
        return has_room and has_check

    def is_action_request(self, question: str) -> bool:
        return (
            self.is_create_request(question)
            or self.is_confirm_request(question)
            or self.is_cancel_request(question)
            or self.is_room_availability_request(question)
        )

    def _rooms_in_question(self, question: str) -> tuple[CalendarRoom, ...]:
        normalized = self._normalize(question)
        selected = []
        for room in self.rooms:
            candidates = (room.id, room.name, *room.aliases)
            if any(self._normalize(candidate) in normalized for candidate in candidates):
                selected.append(room)
        return tuple(selected)

    def _parse_planned_period(
        self,
        plan: dict[str, Any],
        now: datetime,
    ) -> tuple[datetime, datetime]:
        try:
            start = datetime.fromisoformat(str(plan["start"])).astimezone(self.timezone)
            end = datetime.fromisoformat(str(plan["end"])).astimezone(self.timezone)
        except (KeyError, TypeError, ValueError) as exc:
            raise CalendarActionError(
                "Không xác định được ngày giờ. Hãy ghi rõ ngày, giờ bắt đầu và thời lượng."
            ) from exc
        if start <= now:
            raise CalendarActionError("Thời gian bắt đầu phải ở tương lai.")
        if end <= start or end - start > timedelta(hours=8):
            raise CalendarActionError("Thời lượng phải lớn hơn 0 và không quá 8 giờ.")
        return start, end

    async def _check_rooms(self, question: str) -> CalendarActionResult:
        now = datetime.now(self.timezone)
        plan = await self.planner(question, now)
        start, end = self._parse_planned_period(plan, now)
        selected_rooms = self._rooms_in_question(question) or self.rooms
        availability = await self._availability(start, end)
        lines = [
            f"Tình trạng phòng từ {start:%H:%M} đến {end:%H:%M} ngày {start:%d/%m/%Y}:"
        ]
        for room in selected_rooms:
            lines.append(
                f"- {room.name}: {'Còn trống' if availability.get(room.id) else 'Đang bận'}"
            )
        return CalendarActionResult(kind="availability", answer="\n".join(lines))

    def _authorize_owner(self, employee: Any) -> None:
        email = str(getattr(employee, "company_email", "") or "").strip().lower()
        if not email or email != self.owner_email:
            raise CalendarActionError(
                "Chức năng tạo lịch hiện chỉ dành cho chủ tài khoản Google Calendar đã kết nối."
            )

    def _resolve_room(self, room_value: Any) -> CalendarRoom | None:
        if room_value is None or room_value == "":
            return None
        value = self._normalize(str(room_value))
        for room in self.rooms:
            candidates = (room.id, room.name, *room.aliases)
            if any(self._normalize(candidate) == value for candidate in candidates):
                return room
        raise CalendarActionError("Chỉ hỗ trợ Phòng họp 1 hoặc Phòng họp 3.")

    async def _availability(self, start: datetime, end: datetime) -> dict[str, bool]:
        result = await self.tool_runner(
            "check-availability",
            {
                "timeMin": start.isoformat(),
                "timeMax": end.isoformat(),
                "timeZone": str(self.timezone),
                "items": [{"id": room.calendar_id} for room in self.rooms],
            },
        )
        payload = self._tool_payload(result)
        calendars = payload.get("calendars", {})
        return {
            room.id: not bool(calendars.get(room.calendar_id, {}).get("busy", []))
            for room in self.rooms
        }

    @classmethod
    def _tool_payload(cls, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            if "content" in result:
                return cls._tool_payload(result["content"])
            return result
        if isinstance(result, str):
            payload = json.loads(result)
            if not isinstance(payload, dict):
                raise CalendarActionError("Calendar trả về dữ liệu không hợp lệ.")
            return payload
        if isinstance(result, list):
            for block in result:
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    return cls._tool_payload(text)
        raise CalendarActionError("Không đọc được phản hồi từ Google Calendar.")

    def _draft_answer(self, draft: CalendarDraft, availability: dict[str, bool]) -> str:
        lines = [
            "Bản nháp sự kiện:",
            f"- Tiêu đề: {draft.title}",
            f"- Thời gian: {draft.start:%H:%M}–{draft.end:%H:%M}, ngày {draft.start:%d/%m/%Y}",
            f"- Múi giờ: {self.timezone}",
        ]
        if draft.room:
            lines.extend(
                (
                    f"- Phòng: {draft.room.name}",
                    f"- Trạng thái phòng: {'Còn trống' if availability.get(draft.room.id) else 'Đang bận'}",
                )
            )
        else:
            lines.append("- Phòng: Không đặt phòng")
        lines.append("- Người tham gia khác: Không có")
        if draft.room and not availability.get(draft.room.id):
            alternatives = [
                room.name
                for room in self.rooms
                if room.id != draft.room.id and availability.get(room.id)
            ]
            if alternatives:
                lines.append(f"- Phòng còn trống thay thế: {', '.join(alternatives)}")
            lines.append("Phòng đã chọn đang bận nên chưa thể xác nhận tạo lịch.")
        else:
            lines.append('Nhập "Xác nhận tạo lịch" để tạo, hoặc "Hủy bản nháp" để bỏ.')
        return "\n".join(lines)

    async def handle(
        self,
        *,
        session_id: str,
        question: str,
        employee: Any,
    ) -> CalendarActionResult | None:
        if self.is_room_availability_request(question):
            return await self._check_rooms(question)
        if self.is_cancel_request(question):
            self._authorize_owner(employee)
            removed = await self.store.discard(session_id)
            return CalendarActionResult(
                kind="cancelled",
                answer=(
                    "Đã hủy bản nháp sự kiện."
                    if removed
                    else "Hiện không có bản nháp sự kiện nào để hủy."
                ),
            )
        if self.is_confirm_request(question):
            self._authorize_owner(employee)
            return await self._confirm(session_id, employee)
        if not self.is_create_request(question):
            return None

        self._authorize_owner(employee)
        now = datetime.now(self.timezone)
        plan = await self.planner(question, now)
        title = str(plan.get("title") or "").strip()
        if not title:
            raise CalendarActionError("Bạn chưa cung cấp tiêu đề sự kiện.")
        start, end = self._parse_planned_period(plan, now)
        room = self._resolve_room(plan.get("room"))
        availability = await self._availability(start, end)
        draft = CalendarDraft(
            id=str(uuid.uuid4()),
            session_id=session_id,
            employee_id=str(getattr(employee, "id", "")),
            title=title,
            start=start,
            end=end,
            room=room,
        )
        await self.store.put(draft)
        return CalendarActionResult(
            kind="draft",
            answer=self._draft_answer(draft, availability),
            draft=draft,
        )

    async def _confirm(self, session_id: str, employee: Any) -> CalendarActionResult:
        lock = self._confirm_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            draft = await self.store.get(session_id)
            if draft is None:
                return CalendarActionResult(
                    kind="missing",
                    answer="Hiện không có bản nháp sự kiện nào để xác nhận.",
                )
            if draft.employee_id != str(getattr(employee, "id", "")):
                raise CalendarActionError("Bản nháp không thuộc người dùng hiện tại.")
            if draft.status == "created":
                return CalendarActionResult(
                    kind="created",
                    answer=self._created_answer(draft),
                    draft=draft,
                )
            availability = await self._availability(draft.start, draft.end)
            if draft.room and not availability.get(draft.room.id):
                return CalendarActionResult(
                    kind="conflict",
                    answer=(
                        f"{draft.room.name} không còn trống trong khoảng "
                        f"{draft.start:%H:%M}–{draft.end:%H:%M} ngày {draft.start:%d/%m/%Y}. "
                        "Chưa tạo sự kiện."
                    ),
                    draft=draft,
                )
            event: dict[str, Any] = {
                "summary": draft.title,
                "description": "Sự kiện được tạo qua Meibook.",
                "start": {
                    "dateTime": draft.start.isoformat(),
                    "timeZone": str(self.timezone),
                },
                "end": {
                    "dateTime": draft.end.isoformat(),
                    "timeZone": str(self.timezone),
                },
            }
            if draft.room:
                event["attendees"] = [
                    {
                        "email": draft.room.calendar_id,
                        "displayName": draft.room.name,
                    }
                ]
            result = self._tool_payload(
                await self.tool_runner(
                    "create-event",
                    {"calendarId": "primary", "event": event},
                )
            )
            room_status = ""
            if draft.room:
                attendee = next(
                    (
                        item
                        for item in result.get("attendees", [])
                        if item.get("email") == draft.room.calendar_id
                    ),
                    {},
                )
                room_status = str(attendee.get("responseStatus") or "needsAction")
            created = replace(
                draft,
                status="created",
                event_id=str(result.get("id") or ""),
                event_url=str(result.get("htmlLink") or ""),
                room_status=room_status,
            )
            await self.store.put(created)
            return CalendarActionResult(
                kind="created",
                answer=self._created_answer(created),
                draft=created,
            )

    @staticmethod
    def _created_answer(draft: CalendarDraft) -> str:
        answer = (
            f"Đã tạo sự kiện \"{draft.title}\" từ {draft.start:%H:%M} đến "
            f"{draft.end:%H:%M} ngày {draft.start:%d/%m/%Y}."
        )
        if draft.room:
            if draft.room_status == "accepted":
                answer += f" {draft.room.name} đã chấp nhận đặt phòng."
            elif draft.room_status == "declined":
                answer += f" {draft.room.name} đã từ chối yêu cầu đặt phòng."
            else:
                answer += f" Đã gửi yêu cầu tới {draft.room.name}; trạng thái đang chờ xác nhận."
        if draft.event_url:
            answer += f"\nGoogle Calendar: {draft.event_url}"
        return answer

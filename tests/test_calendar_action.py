import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.actions.calendar_action import (
    CalendarActionError,
    CalendarActionService,
    CalendarDraftStore,
)


ROOMS_PATH = Path(__file__).resolve().parents[1] / "config" / "calendar_rooms.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")
OWNER = SimpleNamespace(
    id="000286",
    company_email="son.nguyendinh@meiko.vn",
)
OTHER = SimpleNamespace(
    id="000001",
    company_email="someone.else@meiko.vn",
)


class FakeCalendar:
    def __init__(self, *, busy_room_ids=()):
        self.busy_room_ids = set(busy_room_ids)
        self.calls = []
        self.create_count = 0

    async def run(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if tool_name == "check-availability":
            return {
                "calendars": {
                    item["id"]: {
                        "busy": (
                            [{"start": arguments["timeMin"], "end": arguments["timeMax"]}]
                            if item["id"] in self.busy_room_ids
                            else []
                        )
                    }
                    for item in arguments["items"]
                }
            }
        if tool_name == "create-event":
            self.create_count += 1
            attendee = (arguments["event"].get("attendees") or [{}])[0]
            return {
                "id": "event-123",
                "htmlLink": "https://calendar.google.com/event?eid=fixture",
                "attendees": [
                    {
                        "email": attendee.get("email"),
                        "responseStatus": "accepted",
                    }
                ] if attendee.get("email") else [],
            }
        raise AssertionError(tool_name)


def planner_for(*, room="room-1"):
    async def planner(question, now):
        start = now + timedelta(days=1)
        start = start.replace(hour=9, minute=0, second=0, microsecond=0)
        return {
            "title": "Họp dự án AI",
            "start": start.isoformat(),
            "end": (start + timedelta(hours=1)).isoformat(),
            "room": room,
        }
    return planner


def service(fake, *, room="room-1"):
    return CalendarActionService(
        tool_runner=fake.run,
        planner=planner_for(room=room),
        rooms_path=ROOMS_PATH,
        store=CalendarDraftStore(ttl_seconds=60),
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Đặt phòng họp 1 lúc 9 giờ sáng mai", True),
        ("Tạo lịch họp dự án AI ngày mai", True),
        ("Kiểm tra phòng họp 1 có trống không", False),
        ("Quy định đăng ký lịch làm việc là gì?", False),
    ],
)
def test_create_intent_is_fail_closed(question, expected):
    assert CalendarActionService.is_create_request(question) is expected


def test_room_availability_uses_allowlisted_resource_ids():
    room_1_id = "meiko.vn_18807gio2vmmigs6j37uuje5ivic8@resource.calendar.google.com"
    fake = FakeCalendar(busy_room_ids={room_1_id})
    calendar = service(fake)

    result = asyncio.run(
        calendar.handle(
            session_id="room-check",
            question=(
                "Kiểm tra phòng họp 1 và phòng họp 3 có trống "
                "từ 9 giờ đến 10 giờ sáng mai không?"
            ),
            employee=OWNER,
        )
    )

    assert result.kind == "availability"
    assert "Meeting Room 1_Booking: Đang bận" in result.answer
    assert "Meeting Room 3_Booking: Còn trống" in result.answer
    call = fake.calls[0]
    assert call[0] == "check-availability"
    assert {item["id"] for item in call[1]["items"]} == {
        room.calendar_id for room in calendar.rooms
    }


def test_owner_can_create_draft_and_confirm_room():
    fake = FakeCalendar()
    calendar = service(fake)

    async def run():
        draft = await calendar.handle(
            session_id="session-1",
            question="Đặt phòng họp 1 lúc 9 giờ sáng mai để họp dự án AI",
            employee=OWNER,
        )
        created = await calendar.handle(
            session_id="session-1",
            question="Xác nhận tạo lịch",
            employee=OWNER,
        )
        duplicate = await calendar.handle(
            session_id="session-1",
            question="Xác nhận tạo lịch",
            employee=OWNER,
        )
        return draft, created, duplicate

    draft, created, duplicate = asyncio.run(run())

    assert draft.kind == "draft"
    assert "Còn trống" in draft.answer
    assert created.kind == "created"
    assert "đã chấp nhận đặt phòng" in created.answer
    assert duplicate.answer == created.answer
    assert fake.create_count == 1
    create_call = next(call for call in fake.calls if call[0] == "create-event")
    event = create_call[1]["event"]
    assert "location" not in event
    assert len(event["attendees"]) == 1
    assert event["attendees"][0]["email"].endswith(
        "@resource.calendar.google.com"
    )


def test_room_conflict_prevents_creation():
    room_1_id = "meiko.vn_18807gio2vmmigs6j37uuje5ivic8@resource.calendar.google.com"
    fake = FakeCalendar(busy_room_ids={room_1_id})
    calendar = service(fake)

    async def run():
        draft = await calendar.handle(
            session_id="session-2",
            question="Đặt phòng họp 1 lúc 9 giờ sáng mai",
            employee=OWNER,
        )
        confirm = await calendar.handle(
            session_id="session-2",
            question="Xác nhận tạo lịch",
            employee=OWNER,
        )
        return draft, confirm

    draft, confirm = asyncio.run(run())
    assert "Đang bận" in draft.answer
    assert confirm.kind == "conflict"
    assert fake.create_count == 0


def test_personal_event_has_no_attendees():
    fake = FakeCalendar()
    calendar = service(fake, room=None)

    async def run():
        await calendar.handle(
            session_id="session-3",
            question="Tạo lịch viết báo cáo lúc 9 giờ sáng mai",
            employee=OWNER,
        )
        return await calendar.handle(
            session_id="session-3",
            question="Xác nhận tạo lịch",
            employee=OWNER,
        )

    created = asyncio.run(run())
    create_call = next(call for call in fake.calls if call[0] == "create-event")
    assert "attendees" not in create_call[1]["event"]
    assert created.kind == "created"


def test_non_owner_cannot_create_or_confirm():
    fake = FakeCalendar()
    calendar = service(fake)

    with pytest.raises(CalendarActionError):
        asyncio.run(
            calendar.handle(
                session_id="session-4",
                question="Đặt phòng họp 1 lúc 9 giờ sáng mai",
                employee=OTHER,
            )
        )
    assert fake.calls == []


def test_cancel_and_missing_confirmation_are_safe():
    fake = FakeCalendar()
    calendar = service(fake)

    async def run():
        missing = await calendar.handle(
            session_id="session-5",
            question="Xác nhận tạo lịch",
            employee=OWNER,
        )
        cancelled = await calendar.handle(
            session_id="session-5",
            question="Hủy bản nháp",
            employee=OWNER,
        )
        return missing, cancelled

    missing, cancelled = asyncio.run(run())
    assert "không có bản nháp" in missing.answer
    assert "không có bản nháp" in cancelled.answer
    assert fake.calls == []

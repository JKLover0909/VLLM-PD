"""Định dạng câu trả lời nhân sự (VI/JA) từ dữ liệu đã truy vấn.

Các hàm ở đây thuần túy: nhận dữ liệu đã lấy từ database và trả về chuỗi câu
trả lời. Việc truy vấn database nằm ở ``EmployeeDirectory``. Giữ nguyên hành vi
so với bản gốc.
"""

from typing import Any, Optional

from src.auth.employee_intent import normalize_text


def format_company_headcount(total: int, language: str) -> str:
    if language == "ja":
        return f"社員名簿上、Meiko Automationには現在{total}人が登録されています。"
    return f"Theo danh bạ nhân sự, Meiko Automation hiện có {total} thành viên."


def format_department_catalog(
    summaries: list[dict[str, Any]],
    language: str,
) -> str:
    summaries = sorted(
        summaries,
        key=lambda item: normalize_text(item["department"]),
    )
    total = len(summaries)
    if language == "ja":
        names = "、".join(item["department"] for item in summaries)
        return f"Meiko Automationには合計{total}部門があります。部門は{names}です。"
    names = ", ".join(item["department"] for item in summaries)
    return f"Meiko Automation hiện có tổng cộng {total} phòng ban, gồm: {names}."


def format_largest_departments(
    summaries: list[dict[str, Any]],
    language: str,
) -> Optional[str]:
    if not summaries:
        return None
    max_size = max(item["size"] for item in summaries)
    largest = [item for item in summaries if item["size"] == max_size]
    if language == "ja":
        departments = "、".join(
            f"{item['department']}（{item['size']}人）" for item in largest
        )
        return f"人数が最も多い部門は{departments}です。"
    departments = ", ".join(
        f"{item['department']} ({item['size']} người)" for item in largest
    )
    return f"Phòng ban có nhiều nhân sự nhất là {departments}."


def format_department_counts(
    profiles: list[dict[str, Any]],
    language: str,
) -> str:
    if language == "ja":
        if len(profiles) == 1:
            profile = profiles[0]
            return f"{profile['department']}部門には{profile['size']}人が在籍しています。"
        lines = [
            f"- {profile['department']}: {profile['size']}人"
            for profile in profiles
        ]
        return "該当する部門の人数は以下の通りです。\n" + "\n".join(lines)

    if len(profiles) == 1:
        profile = profiles[0]
        return f"Phòng {profile['department']} hiện có {profile['size']} người."
    lines = [
        f"- {profile['department']}: {profile['size']} người"
        for profile in profiles
    ]
    return "Số nhân sự của các phòng ban được hỏi là:\n" + "\n".join(lines)


def format_department_leadership(
    profiles: list[dict[str, Any]],
    language: str,
) -> str:
    lines: list[str] = []
    for profile in profiles:
        leaders = profile["heads"] + profile["deputies"]
        if language == "ja":
            if leaders:
                lines.append(f"{profile['department']}: " + "、".join(leaders))
            else:
                lines.append(f"{profile['department']}: 管理者情報は登録されていません。")
        else:
            if leaders:
                lines.append(f"- {profile['department']}: " + "; ".join(leaders))
            else:
                lines.append(
                    f"- {profile['department']}: chưa có thông tin trưởng/phó phòng."
                )
    if language == "ja":
        return "該当部門の管理者情報は以下の通りです。\n" + "\n".join(lines)
    return "Thông tin trưởng/phó phòng là:\n" + "\n".join(lines)


def format_department_rosters(
    profiles: list[dict[str, Any]],
    language: str,
) -> str:
    sections: list[str] = []
    for profile in profiles:
        members = profile["members"]
        if language == "ja":
            lines = [
                f"{member['name']}（{member['position'] or '未登録'}）"
                for member in members
            ]
            sections.append(
                f"{profile['department']}部門（{profile['size']}人）:\n"
                + "\n".join(lines)
            )
        else:
            lines = [
                f"- {member['name']} ({member['position'] or 'chưa có chức danh'})"
                for member in members
            ]
            sections.append(
                f"Phòng {profile['department']} ({profile['size']} người):\n"
                + "\n".join(lines)
            )
    return "\n\n".join(sections)


def format_people_profiles(
    people: list[dict[str, Any]],
    language: str,
) -> str:
    if language == "ja":
        lines = []
        for person in people:
            line = (
                f"- {person['name']}（社員番号: {person['id']}、"
                f"部門: {person.get('department') or '未登録'}、"
                f"役職: {person.get('position') or '未登録'}"
            )
            if person.get("birth_date"):
                line += f"、生年月日: {person['birth_date']}"
            if person.get("marital_status"):
                line += f"、婚姻状況: {person['marital_status']}"
            lines.append(line + "）")
        return "該当する社員情報は以下の通りです。\n" + "\n".join(lines)
    lines = []
    for person in people:
        line = (
            f"- {person['name']} - mã nhân viên {person['id']}, "
            f"phòng {person.get('department') or 'chưa có'}, "
            f"chức danh {person.get('position') or 'chưa có'}"
        )
        if person.get("birth_date"):
            line += f", ngày sinh {person['birth_date']}"
        if person.get("marital_status"):
            line += f", tình trạng hôn nhân: {person['marital_status']}"
        lines.append(line)
    return "Thông tin nhân sự tìm thấy:\n" + "\n".join(lines)

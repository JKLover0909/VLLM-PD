from src.auth.employee_directory import EmployeeDirectory, GUEST_EMPLOYEE_ID


def test_guest_employee_profile_does_not_require_database(tmp_path):
    directory = EmployeeDirectory(tmp_path / "missing.sqlite")

    guest = directory.profile(GUEST_EMPLOYEE_ID)

    assert guest is not None
    assert guest["id"] == GUEST_EMPLOYEE_ID
    assert guest["name"] == "Guest"
    assert guest["company_email"] == ""
    assert guest["greeting"] == "Chào mừng đến với hệ thống Meibook,"


def test_unknown_employee_still_requires_directory_database(tmp_path):
    directory = EmployeeDirectory(tmp_path / "missing.sqlite")

    assert directory.profile("123456") is None


def test_safe_short_name_alias_phi_maps_to_nguyen_trong_phi():
    directory = EmployeeDirectory("data/employee_directory.sqlite")

    people = directory.people_context_for_question("Anh Phi làm phòng nào?")

    assert people
    assert people[0]["name"] == "Nguyễn Trọng Phi"
    assert people[0]["department"] == "ICT"

from src.auth.employee_directory import EmployeeDirectory, GUEST_EMPLOYEE_ID


def test_guest_employee_profile_does_not_require_database(tmp_path):
    directory = EmployeeDirectory(tmp_path / "missing.sqlite")

    guest = directory.profile(GUEST_EMPLOYEE_ID)

    assert guest is not None
    assert guest["id"] == GUEST_EMPLOYEE_ID
    assert guest["name"] == "Guest"
    assert guest["greeting"] == "Chào mừng đến với hệ thống Meibook,"


def test_unknown_employee_still_requires_directory_database(tmp_path):
    directory = EmployeeDirectory(tmp_path / "missing.sqlite")

    assert directory.profile("123456") is None

import pytest

from calculator import add, divide


def test_add_returns_sum():
    assert add(2, 3) == 5


def test_divide_returns_quotient():
    assert divide(10, 2) == 5


def test_divide_rejects_zero_denominator():
    with pytest.raises(ValueError, match="must not be zero"):
        divide(10, 0)

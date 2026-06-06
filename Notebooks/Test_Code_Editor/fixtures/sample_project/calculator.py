"""Small calculator fixture used to evaluate coding agents."""


def add(a: float, b: float) -> float:
    return a + b


def divide(a: float, b: float) -> float:
    # Intentional Milestone 1 bug: no explicit zero-division validation.
    return a / b

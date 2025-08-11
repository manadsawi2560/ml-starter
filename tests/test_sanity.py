from ml_starter.utils import add


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(2.5, 2.5) == 5.0
    assert add(-2, -3) == -5

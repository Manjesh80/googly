from ..arrays import *


def test_separate_even_odd():
    arr = [*range(10)]
    separate_even_odd(arr)
    assert arr[0] % 2 == 0
    assert arr[1] % 2 == 0
    assert arr[2] % 2 == 0

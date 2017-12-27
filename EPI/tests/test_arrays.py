from ..arrays import *


def test_separate_even_odd():
    arr = [*range(10)]
    separate_even_odd(arr)
    assert arr[0] % 2 == 0
    assert arr[1] % 2 == 0
    assert arr[2] % 2 == 0


def test_multiplication():
    arr = multiply_two_arrays([1, 2], [1, 3])
    print(arr)
    assert arr[0] == 0
    assert arr[1] == 1
    assert arr[2] == 5
    assert arr[3] == 6

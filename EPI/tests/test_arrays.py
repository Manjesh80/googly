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


# 5.9 - Enumerate all the primes of N
# test command ==> pytest EPI\tests\test_arrays.py::test_generate_prime
def test_generate_prime():
    res = generate_prime(100)
    assert len(res) == 25
    assert res.count(96) == 1


# 5.10 - Permute the elements of a Array
# test command ==> pytest EPI\tests\test_arrays.py::test_permute_array_with_o1_space
def test_permute_array_with_o1_space():
    res = permute_array_with_o1_space(['a', 'b', 'c', 'd'], [2, 0, 1, 3])
    assert len(res) == 4
    assert res[0] == 'b'
    assert res[1] == 'c'
    assert res[2] == 'a'
    assert res[3] == 'd'


# 5.11 - Compute the next permutation
# test command ==> pytest EPI\tests\test_arrays.py::test_compute_next_permutation
def test_compute_next_permutation():
    res = compute_next_permutation([6, 2, 3, 5, 4, 1, 0])
    assert res == [6, 2, 4, 0, 1, 3, 5]
    pass

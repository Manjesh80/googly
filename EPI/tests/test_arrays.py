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


# 5.1 - Dutch national flag
# pytest EPI\tests\test_arrays.py::test_dutch_national_flag
def test_dutch_national_flag():
    A = [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]
    dutch_national_flag(A, 1)
    assert A[0] == 0
    assert A[4] == 1
    assert A[8] == 2


# 5.2 - Increment an arbitrary precision integer
# pytest EPI\tests\test_arrays.py::test_increment_integer
def test_increment_integer():
    A = [1, 2, 9]
    increment_integer(A)
    assert A[2] == 0


# 5.3 Multiply arbitrary integer
# pytest EPI\tests\test_arrays.py::test_multiply_two_arrays
def test_multiply_two_arrays():
    res = multiply_two_arrays([1, 2], [1, 2])
    print(res)
    assert res[0] == 0
    assert res[1] == 1
    assert res[2] == 4
    assert res[3] == 4


# 5.4 - Advancing through an array , whether you can jump and go to end
# pytest EPI\tests\test_arrays.py::test_advance_through_an_array
def test_advance_through_an_array():
    assert True == advance_through_an_array([3, 3, 0, 0, 2, 0, 1])
    assert False == advance_through_an_array([3, 2, 0, 0, 2, 0, 1])


# 5.5 - Delete duplicates from sorted array
# pytest -s EPI\tests\test_arrays.py::test_del_duplicate_from_sorted_array
def test_del_duplicate_from_sorted_array():
    A = [2, 3, 5, 5, 5, 7, 11, 11, 11, 13]
    res = del_duplicate_from_sorted_array(A)
    assert len(res) == 6


# 5.7 - Buy and sell once
# pytest -s EPI\tests\test_arrays.py::test_buy_and_sell_only_once
def test_buy_and_sell_only_once():
    A = [2, 2, 5, 4, 1, 20]
    res = buy_and_sell_only_once(A)
    assert res[0] == 19


# 5.7 - Buy and sell stock twice
# pytest -s EPI\tests\test_arrays.py::test_buy_and_sell_only_twice
def test_buy_and_sell_only_twice():
    A = [2, 2, 5, 4, 1, 20]
    res = buy_and_sell_only_twice(A)
    assert res == 22


# 5.8 - Computing and alternation a <= b >= c <= d >= e
# pytest -s EPI\tests\test_arrays.py::test_rearrange
def test_rearrange():
    A = [2, 1, 0, 6, 8, 9, 4, 3, 10, 5]
    rearrange(A)
    assert (A[0] <= A[1] >= A[2])


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


# 5.12 - Sample offline data
# test command ==> pytest EPI\tests\test_arrays.py::test_generate_random_offline_data
def test_generate_random_offline_data():
    res = generate_random_offline_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
    assert len(res) == 3


# 5.13 - Sample online data
# test command ==> pytest -s EPI\tests\test_arrays.py::test_generate_random_sample_online
def test_generate_random_sample_online():
    sample = generate_random_sample_online(ordered_seq_generator(), 10)
    assert len([x for x in sample if x > 50]) > 2


# 5.14 - Compute a Random permutation
# pytest -s EPI\tests\test_arrays.py::test_generate_random_permutation
def test_generate_random_permutation():
    sample = generate_random_permutation([10, 20, 30, 40, 50, 60])
    assert len(sample) == 6


# 5.16 - Generate non-uniform random numbers
# pytest -s EPI\tests\test_arrays.py::test_generate_non_uniform_random_numbers
def test_generate_non_uniform_random_numbers():
    probability = [0.05, 0.1, 0.15, 0.20, 0.5]
    res = Counter()
    for i in range(100):
        res[generate_non_uniform_random_numbers([10, 20, 30, 40, 50], probability)] += 1
    assert res[50] >= 50

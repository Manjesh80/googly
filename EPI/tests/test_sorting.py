from ..sorting import *


# 13.1 Compute the intersection of the sorted array
# pytest -s EPI\tests\test_sorting.py::test_compute_intersection_of_sorted_array
def test_compute_intersection_of_sorted_array():
    A = [2, 3, 3, 5, 5, 6, 7, 7, 8, 12]
    B = [5, 5, 6, 8, 8, 9, 10]
    res = compute_intersection_of_sorted_array(A, B)
    assert len(res) == 3


# 13.2 merge_two_sorted_arrays_with_blank_space
# pytest -s EPI\tests\test_sorting.py::test_merge_two_sorted_arrays_with_blank_space
def test_merge_two_sorted_arrays_with_blank_space():
    A = [2, 3, 5, 7, None, None, None, None]
    B = [1, 4, 6, 8]
    res = merge_two_sorted_arrays_with_blank_space(A, B)
    assert res[0] == 1
    assert res[1] == 2


# 13.3 Remove first name duplicates
# pytest -s EPI\tests\test_sorting.py::test_remove_first_name_duplicates
def test_remove_first_name_duplicates():
    students = [Student('D', 5), Student('A', 30), Student('A', 35), Student('B', 20), Student('C', 10)]
    remove_first_name_duplicates(students)
    assert len([std for std in students if std.name == 'A']) == 1


# 13.4 Smallest non constructable value
# pytest -s EPI\tests\test_sorting.py::test_small_non_constructable_value
def test_small_non_constructable_value():
    change = [1, 1, 1, 1, 1, 5, 10, 25]
    res = small_non_constructable_value(change)
    assert res == 21

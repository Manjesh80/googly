from ..heaps import *


# 10.1 Merge sorted array
# pytest -s EPI\tests\test_heaps.py::test_merge_sorted_array
def test_merge_sorted_array():
    res = merge_sorted_array([[2, 20, 200], [3, 30, 300], [4, 40, 400]])
    assert res[0] == 2
    assert res[-1:] == [400]


# 10.2 Sort K ascending and descending
# pytest -s EPI\tests\test_heaps.py::test_sort_k_increasing_decresing_array
def test_sort_k_increasing_decresing_array():
    l = [57, 131, 493, 294, 221, 339, 418, 452, 442, 190]
    sorted_array = merge_sorted_array(sort_k_increasing_decresing_array(l))
    print(sorted_array)
    assert len(sorted_array) == 10

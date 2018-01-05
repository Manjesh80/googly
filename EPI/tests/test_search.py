from ..search import *


# 10.1 Find first occurrence of where array has duplicates
# pytest -s EPI\tests\test_search.py::test_find_first_occurrence_k_has_duplicates
def test_find_first_occurrence_k_has_duplicates():
    sorted_array = [-14, -10, 2, 108, 108, 243, 285, 285, 285, 401]
    assert 6 == find_first_occurrence_k_has_duplicates(sorted_array, 285)


# 10.2 Search a sorted array to see if any matching index
# pytest -s EPI\tests\test_search.py::test_find_first_occurrence_k_has_duplicates
def test_search_a_sorted_array_for_entry_equal_to_index():
    sorted_array = [-2, 0, 2, 3, 2, 2, 3]
    assert 3 == search_a_sorted_array_for_entry_equal_to_index(sorted_array)

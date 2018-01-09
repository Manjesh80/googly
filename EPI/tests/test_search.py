from ..search import *


# 11.1 Find first occurrence of where array has duplicates
# pytest -s EPI\tests\test_search.py::test_find_first_occurrence_k_has_duplicates
def test_find_first_occurrence_k_has_duplicates():
    sorted_array = [-14, -10, 2, 108, 108, 243, 285, 285, 285, 401]
    assert 6 == find_first_occurrence_k_has_duplicates(sorted_array, 285)


# 11.2 Search a sorted array to see if any matching index
# pytest -s EPI\tests\test_search.py::test_find_first_occurrence_k_has_duplicates
def test_search_a_sorted_array_for_entry_equal_to_index():
    sorted_array = [-2, 0, 2, 3, 2, 2, 3]
    assert 3 == search_a_sorted_array_for_entry_equal_to_index(sorted_array)


# 11.3 Search a cyclically sorted array
# This is bit of crap logic, so thinking a simple is always better
# pytest -s EPI\tests\test_search.py::test_search_starting_point_of_cyclic_sorted_array
def test_search_starting_point_of_cyclic_sorted_array():
    assert 4 == search_starting_point_of_cyclic_sorted_array([38, 478, 550, 631, 103, 203, 220, 234, 279, 368])


# 11.4 Compute nearest integer square root
# Think better and write the algorithm
# pytest -s EPI\tests\test_search.py::test_compute_nearest_integer_square_root
def test_compute_nearest_integer_square_root():
    assert 17 == compute_nearest_integer_square_root(300)

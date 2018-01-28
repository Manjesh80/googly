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


# 13.5 Number of simultaneous events
# pytest -s EPI\tests\test_sorting.py::test_num_of_concurrent_interviews
def test_num_of_concurrent_interviews():
    meetings = [(1, 4), (2, 4), (3, 4), (4, 5)]
    res = num_of_concurrent_interviews(meetings)
    assert res == 3


# 13.6 Merge interval
# pytest -s EPI\tests\test_sorting.py::test_merge_interval
def test_merge_interval():
    meetings = [(0, 2), (3, 6), (7, 9), (11, 12), (14, 17)]
    res = merge_interval(meetings, (1, 8))
    assert res[0][0] == 0
    assert res[0][1] == 9


# pytest -s EPI\tests\test_sorting.py::test_heap_pop
def test_heap_pop():
    A = [5, 4, 8, 2, 7, 10, 3, 6, 1, 9]
    assert heap_pop(A) == 1
    assert heap_pop(A) == 2
    assert heap_pop(A) == 3
    assert heap_pop(A) == 4
    assert heap_pop(A) == 5


# 13.7 Compute union of intervals
# pytest -s EPI\tests\test_sorting.py::test_union_interval
def test_union_interval():
    intervals = [(0, 3), (3, 4), (2, 4), (1, 1),
                 (6, 7), (7, 8), (8, 11), (9, 11),
                 (12, 14), (12, 16), (13, 15), (16, 17)]
    res = union_interval(intervals)
    assert len(res) == 3


# 13.8 sorted_keys_together
# pytest -s EPI\tests\test_sorting.py::test_sort_key_together_epi
def test_sort_key_together_epi():
    students = [Person(age=v.split(',')[0], name=v.split(',')[1]) for v in "1,A#2,B#3,C#1,D#2,E#3,F".split("#")]
    sort_key_together_epi(students)
    assert students[1][0] == '1'


# 13.9 Team photo day 1
# pytest -s EPI\tests\test_sorting.py::test_team_photo_day_one
def test_team_photo_day_one():
    t1 = [1, 2, 3]
    t2 = [6, 5, 7]
    assert True == team_photo_day_one(t1, t2)


# 13.10 Stable sort list
# pytest -s EPI\tests\test_sorting.py::test_stable_sort_list
def test_stable_sort_list():
    dummy_head = tail = ListNode()
    for i in reversed(range(0, 10)):
        tail.next_node = ListNode(i)
        tail = tail.next_node
    sorted_node = stable_sort_list(dummy_head.next_node)
    assert sorted_node.data == 0


# 13.11 Compute Salary threshold
# pytest -s EPI\tests\test_sorting.py::test_find_salary_cap
def test_find_salary_cap():
    res = find_salary_cap(210, [20, 30, 40, 90, 100])
    assert res == 60

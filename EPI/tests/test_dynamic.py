from ..dynamic import *


# 16.X Rod cutting problem
# pytest EPI\tests\test_dynamic.py::test_optimize_rod_cut
def test_optimize_rod_cut():
    table = defaultdict(None)
    table[1] = 2
    table[2] = 5
    table[3] = 4
    table[4] = 9
    table[6] = 14
    res = optimize_rod_cut(table, 6)
    assert res == 15


# 16.1 different score combinations
# pytest -s EPI\tests\test_dynamic.py::test_num_comb_final_score
def test_num_comb_final_score():
    res = num_comb_final_score(6, [1, 2, 3])
    # res = num_comb_final_score(12, [2, 3, 7])
    assert res == 7


# 16.2 Minimum edit distance
# pytest -s EPI\tests\test_dynamic.py::test_minimum_edit_distance
def test_minimum_edit_distance():
    res = minimum_edit_distance(" KITTEN", " KNITTING")
    assert res[-1][-1][0] == 3


# 16.3 Different ways to traverse Matrix
# pytest EPI\tests\test_dynamic.py::test_traverse_2d_array
def test_traverse_2d_array():
    # A = [[*range(i, i + 10)] for i in range(1, 100, 10)]
    A = [[*range(i, i + 3)] for i in range(1, 9, 3)]
    paths = traverse_2d_array(A, 0, 0)
    assert len(paths) == 6
    # assert len(paths) == 48620


# 16.6 Knapsack problem
# pytest -s EPI\tests\test_dynamic.py::test_knapsack_value
def test_knapsack_value():
    items = [(5, 60), (3, 50), (4, 70), (2, 30)]
    res = knapsack_value(items, 5)
    print(res)
    assert res[3][5] == 80
    # items = [(5, 5), (6, 4), (8, 7), (4, 7)]
    # res = knapsack_value_updated(items, 13)
    # final res 14


# 16.9 pickup coins for maximum gain
# pytest -s EPI\tests\test_dynamic.py::test_pick_up_coins_fox_max
def test_pick_up_coins_fox_max():
    res = pick_up_coins_fox_max([10, 25, 5, 1, 10, 5])
    assert res[0][-1] == 31


# 16.10 Number of moves to climb the stairs
# pytest -s EPI\tests\test_dynamic.py::test_num_of_moves_to_climb_stairs
def test_num_of_moves_to_climb_stairs():
    res = num_of_moves_to_climb_stairs(4, 3)
    assert res == 9

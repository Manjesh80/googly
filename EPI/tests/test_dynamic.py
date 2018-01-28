from ..dynamic import *


# 16.3 Different ways to traverse Matrix
# pytest EPI\tests\test_dynamic.py::test_traverse_2d_array
def test_traverse_2d_array():
    # A = [[*range(i, i + 10)] for i in range(1, 100, 10)]
    A = [[*range(i, i + 3)] for i in range(1, 9, 3)]
    paths = traverse_2d_array(A, 0, 0)
    assert len(paths) == 6
    # assert len(paths) == 48620


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

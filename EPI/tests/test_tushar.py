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
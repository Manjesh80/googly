from ..parallel import *


# 19.3 Implement synchronization for Even-Odd interleaved thread
# pytest -s EPI\tests\test_parallel.py::test_print_even_odd
def test_print_even_odd():
    res = print_even_odd(5)
    print(res)
    assert res == [i for i in range(1, 11)]


# 19.51 Understand re-entrant lock
# pytest -s EPI\tests\test_parallel.py::test_demo_re_entrant_lock
def test_demo_re_entrant_lock():
    demo_re_entrant_lock()
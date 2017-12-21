from ..primitive_types import *


def test_reverse_num():
    assert reverse_num(32768) == 1
    assert reverse_num(49152) == 3

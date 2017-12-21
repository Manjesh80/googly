from ..primitive_types import *


def test_swap_bits():
    assert swap_bits(int(0b1001001), 6, 1) == int(0b1011)

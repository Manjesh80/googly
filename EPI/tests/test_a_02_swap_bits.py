from ..primitive_types import *


def test_swap_bits():
    assert swap_bits(int(0b1001001), 6, 1) == int(0b1011)


def test_swap_bits_2():
    x = swap_bits(int(0b1000000000000000), 15, 0)


def test_04_03_reverse_bits():
    input = 0b0000000000000000000000000000000000000000000000001110000000000001
    output = 0b1000000000000111000000000000000000000000000000000000000000000000
    assert reverse_bits(input) == output
    assert reverse_bits(output) == input

from ..primitive_types import *


# Common

def test_get_value_of_last_bit():
    assert get_value_of_last_bit(1) == 1
    assert get_value_of_last_bit(2) == 0


def test_calculate_parity_by_number_of_ones():
    assert calculate_parity_by_number_of_ones(8) == 1
    assert calculate_parity_by_number_of_ones(6) == 0


def test_calculate_parity_by_fixed():
    parity_dict = build_parity_dict(MASK_16_BIT)
    assert calculate_parity_fixed(8, parity_dict) == 1
    assert calculate_parity_fixed(6, parity_dict) == 0


def test_calculate_parity_just_mug():
    assert calculate_parity_just_mug(8) == 1
    assert calculate_parity_just_mug(6) == 0


def test_right_propogate_right_most_to_one_in_O1():
    assert right_propogate_right_most_to_one_in_O1(80) == 95


def test_build_parity_dict():
    parity_dict = build_parity_dict(MASK_16_BIT)

    assert parity_dict[0] == 0
    assert parity_dict[1] == 1
    assert parity_dict[8] == 1
    assert parity_dict[6] == 0
    assert parity_dict[10] == 0


def test_swap_bits():
    assert swap_bits(int(0b1001001), 6, 1) == int(0b1011)


def test_reverse_num():
    assert reverse_num(32768) == 1
    assert reverse_num(49152) == 3


def test_swap_bits_2():
    x = swap_bits(int(0b1000000000000000), 15, 0)


def test_04_03_reverse_bits():
    input = 0b0000000000000000000000000000000000000000000000001110000000000001
    output = 0b1000000000000111000000000000000000000000000000000000000000000000
    assert reverse_bits(input) == output
    assert reverse_bits(output) == input


def test_04_08_reverse_number():
    assert reverse_number(413) == 314
    assert reverse_number(12345) == 54321



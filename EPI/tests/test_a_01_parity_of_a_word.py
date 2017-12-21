from ..primitive_types import *


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


def test_build_parity_dict():
    parity_dict = build_parity_dict(MASK_16_BIT)

    assert parity_dict[0] == 0
    assert parity_dict[1] == 1
    assert parity_dict[8] == 1
    assert parity_dict[6] == 0
    assert parity_dict[10] == 0

#  If a number has odd number of 1's the parity in 1 else zero

from .a_00_common import *


def calculate_parity_by_number_of_ones(x):
    num_of_1 = 0
    while x:
        y = x & (~(x - 1))
        x = x - y
        num_of_1 += 1
    return num_of_1 % 2


def build_parity_dict(mask=0xFFFF):
    return {x: calculate_parity_by_number_of_ones(x) for x in range(0, int(mask))}


def calculate_parity_fixed(x, parity_dict, mask_size_bit=16, mask=MASK_16_BIT):
    """
    Mask with 16 bits at a time, and do a lookup.
    This is great for massive lookup like discussion with lalit on storing 2**64 combinations
    :param x:
    :param parity_dict:
    :param mask_size_bit:
    :param mask:
    :return:
    """
    return (parity_dict[x & mask] ^
            parity_dict[x >> mask_size_bit & mask] ^
            parity_dict[x >> (2 * mask_size_bit) & mask] ^
            parity_dict[x >> (3 * mask_size_bit) & mask])


def calculate_parity_just_mug(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 0x1


def right_propogate_right_most_to_one_in_O1(x):
    """
    Right propagate rught most bit to one in O(1)
    # Use this logc to get the first set 1 bit and reduce by 1, this will give all 1's for rest
    then do XOR to get the desired result
    example
    01010000  to 01011111
    :param x:
    :return:
    """
    val = x & (~(x - 1))
    return x ^ (val - 1)


# TODO page 27 variant , find 77 mod 64

if __name__ == "__main__":
    right_propogate_right_most_to_one_in_O1(80)

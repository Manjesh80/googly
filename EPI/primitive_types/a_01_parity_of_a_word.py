#  If a number has odd number of 1's the parity in 1 else zero

MASK_04_BIT = 0x3
MASK_04_BIT = 0x7
MASK_08_BIT = 0xFF
MASK_16_BIT = 0xFFFF
MASK_32_BIT = 0xFFFFFFFF
MASK_64_BIT = 0xFFFFFFFFFFFFFFFF


def is_64_bit(x):
    '''
    :param x:
    :return:
    If you mask with 64 bit it should give the same value the its 64 bit , test case
    pass 2**40 i.e.  1099511627776
    is_64_bit(1099511627776) // returns true
    '''
    return True if x == (x & MASK_64_BIT) else False


def calculate_parity_by_number_of_ones(x):
    num_of_1 = 0
    while x:
        y = x & (~(x - 1))
        x = x - y
        num_of_1 += 1
    return num_of_1 % 2


def build_parity_dict(mask=0xFFFF):
    return {x: calculate_parity_by_number_of_ones(x) for x in range(0, int(mask))}


def build_dict_mask_by_power():
    """
    2**2 = 11 i.e.4  ( 4-1 )
    2**3 = 111 i.e.7 ( 8-1)
    2**4 = 1111 i.e.15
    :return:
    """
    return {x: (2 ** x) - 1 for x in range(0, 64)}


def calculate_parity_fixed(x, parity_dict, mask_size_bit=16, mask=0xFFFF):
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

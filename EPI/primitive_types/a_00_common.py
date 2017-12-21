MASK_04_BIT = 0x3
MASK_04_BIT = 0x7
MASK_08_BIT = 0xFF
MASK_16_BIT = 0xFFFF
MASK_32_BIT = 0xFFFFFFFF
MASK_64_BIT = 0xFFFFFFFFFFFFFFFF


def build_dict_mask_by_power():
    """
    2**2 = 11 i.e.4  ( 4-1 )
    2**3 = 111 i.e.7 ( 8-1)
    2**4 = 1111 i.e.15
    :return:
    """
    return {x: (2 ** x) - 1 for x in range(0, 64)}


def is_64_bit(x):
    """
    :param x:
    :return:
    If you mask with 64 bit it should give the same value the its 64 bit , test case
    pass 2**40 i.e.  1099511627776
    is_64_bit(1099511627776) // returns true
    """
    return True if x == (x & MASK_64_BIT) else False


def swap_bits(x, i, j):
    if i <= j:
        raise ValueError()
    if ((x >> i) & 1) != ((x >> j) & 1):
        mask = (1 << i) | (1 << j)
        x = x ^ mask

    return x


def reverse_num(x):
    i_cur = 15
    j_cur = 0
    for i in range(0, 8):
        x = swap_bits(x, i_cur, j_cur)
        i_cur = i_cur - 1
        j_cur = j_cur + 1
    return x


def build_reverse_dict_16():
    return {i: reverse_num(i) for i in range(0, 2 ** 16 - 1)}


def reverse_bits(x, mask_size=16, mask=MASK_16_BIT):
    reverse_dict = build_reverse_dict_16()
    return ((reverse_dict[x & mask] << (3 * mask_size)) |
            (reverse_dict[(x >> mask_size) & mask] << (2 * mask_size)) |
            (reverse_dict[(x >> 2 * mask_size) & mask] << mask_size) |
            (reverse_dict[(x >> 3 * mask_size) & mask]))


if __name__ == "__main__":
    val = reverse_bits(0b1110000000000001)
    print(f"Output is ==> {bin(val)}")

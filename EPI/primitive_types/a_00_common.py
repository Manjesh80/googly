import math
import random
from collections import namedtuple

MASK_04_BIT = 0x3
MASK_04_BIT = 0x7
MASK_08_BIT = 0xFF
MASK_16_BIT = 0xFFFF
MASK_32_BIT = 0xFFFFFFFF
MASK_64_BIT = 0xFFFFFFFFFFFFFFFF

UNSIGNED_INT_LENGTH = 64


# commom

# Get value of last bit
def get_value_of_last_bit(x):
    return x & 1


# Add 2 numbers by bit
def add_numbers(x, y):
    carry_on = 0
    result = 0
    dict_by_power = build_dict_by_power()
    for i in range(0, UNSIGNED_INT_LENGTH):
        x_bit, y_bit = get_value_of_last_bit(x >> i), get_value_of_last_bit(y >> i)
        set_bit = 0
        if x_bit & y_bit:
            set_bit, carry_on = (1, 1) if carry_on else (0, 1)
        elif x_bit | y_bit:
            set_bit = 0 if carry_on else 1
        elif not (x_bit | y_bit):
            set_bit, carry_on = (1, 0) if carry_on else (0, 0)

        if set_bit:
            mask = dict_by_power[i]
            result = result ^ mask
    return result


# 4.5 Add two numbers without *
def multiply(a, b):
    running_sum = 0
    while a:
        if a & 1:
            running_sum = add_numbers(running_sum, b)
        a, b = a >> 1, b << 1
    return running_sum


# 4.6 Divide numbers
def divide(x, y):
    result, power = 0, 64
    y_power = y << 64
    while x >= y:
        while y_power > x:
            y_power >>= 1
            power -= 1
        result += 1 << power
        x -= y_power
    return result


# 4.7 X to the power of Y
def x_power_y(x, y):
    result, power = 1.0, y
    if y < 0:
        x, power = 1 / x, -power

    while power:
        if power & 1:
            result *= x
        x, power = x * x, power >> 1
    return result


# 4.8 reverse a number
def reverse_number(x):
    result, x_remaining = 0, abs(x)
    while x_remaining:
        result = result * 10 + x_remaining % 10;
        x_remaining //= 10
    return result if x > 0 else -result


# 4.9 is number a palindrome
def is_number_a_palindrome(x):
    num_of_digits = math.floor(math.log10(x)) + 1
    mask_digit = 10 ** (num_of_digits - 1)
    result = True
    for i in range(num_of_digits // 2):
        head = x // mask_digit
        tail = x % 10
        if head != tail:
            result = False
            break
        x %= mask_digit
        x //= 10
        mask_digit //= 100
    return result;


# 4.10 generate a uniform random number
def generate_uniform_random_number(lb, ub):
    # Example scenario 1, 6 in dice
    num_of_outcomes = ub - lb + 1
    while True:
        result, i = 0, 0
        while (1 << i) < num_of_outcomes:  # if number of outcomes is 6 then it moved 3 times  1,2,4,8
            result = result << 1 | random.choices([0, 1])[0]
            i += 1
        if result < num_of_outcomes:  # here if all comes as 111 the value will be 7 , but our range is 1 to 6 so retry
            break
    return result + lb


Rectangle = namedtuple("Rectangle", ("x", "y", "w", "h"))


# 4.11 Intersection of rectangles white board : Page 35
def do_rectangle_intersect():
    pass


def build_dict_mask_by_power():
    """
    2**2 = 11 i.e.4  ( 4-1 )
    2**3 = 111 i.e.7 ( 8-1)
    2**4 = 1111 i.e.15
    :return:
    """
    return {x: (2 ** x) - 1 for x in range(0, 64)}


def build_dict_by_power():
    """
    2**2 = 11 i.e.4  ( 4-1 )
    2**3 = 111 i.e.7 ( 8-1)
    2**4 = 1111 i.e.15
    :return:
    """
    return {x: (2 ** x) for x in range(0, 64)}


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


def count_nume_of_bits(x):
    bit_count = 0;
    while x:
        bit_count += x & 1
        x >>= 1
    return bit_count


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


if __name__ == "__main__":
    val = is_number_a_palindrome(12345321)
    print(f"Output is ==> {val}")

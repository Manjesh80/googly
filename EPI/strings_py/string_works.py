from collections import OrderedDict
from functools import reduce

Mapping = ['0', '1', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ']


# Mapping = ['0', '1', 'AB', 'CD']


def mnemonic(digit):
    res = []
    for c in Mapping[digit]:
        res.append(c)
    return res


def cross_the_list(a, b):
    res = []
    if a is None or b is None:
        return a or b
    for item_a in a:
        for item_b in b:
            res.append(item_a + item_b)
    return res


def phone_memonic(phone_number):
    def phone_memonic_helper(digit):
        if digit == len(phone_number):
            mnemonics.append(''.join(partial_mnemonic))
        else:
            for c in Mapping[int(phone_number[digit])]:
                partial_mnemonic[digit] = c
                phone_memonic_helper(digit + 1)

    mnemonics, partial_mnemonic = [], [0] * len(phone_number)
    phone_memonic_helper(0)
    return mnemonics


# 6.7 Find all mnemonics of a phone number
def phone_combination(number):
    def build_combination(digit):
        if digit == len(number):
            combinations.append("".join(combination))
        else:
            for alpha in Mapping[int(number[digit])]:
                combination[digit] = alpha
                build_combination(digit + 1)

    combinations, combination = [], [0] * len(number)
    build_combination(0)
    return combinations


# 6.9 Convert from Roman to Decimal
# I ==> V, X
# X ==> L, C
# C ==> M , D


ROMANS = {'I': (1, 1), 'V': (2, 5),
          'X': (3, 10), 'L': (4, 50),
          'C': (5, 100), 'D': (6, 500), 'M': (7, 1000)}


def roman_to_decimal(roman):
    result = 0
    precedence_index = OrderedDict([5, 3, 1])
    current_highest_index_allowed = precedence_index.pop()
    for alpha in roman:
        current_roman = ROMANS[alpha]
        pass
    return result


def to_int(val):
    try:
        return int(val)
    except ValueError:
        return -1


def is_valid_part(part):
    part = to_int(part)
    return True if 0 <= part <= 255 else False


# 6.10 Compiute all valid IP address
def get_valid_ip_addresses(ip_string):
    valid_ip_addresses = []
    for i in range(1, 4):
        part1 = ip_string[0: i]
        if not is_valid_part(part1):
            continue
        for j in range(1, 4):
            part2 = ip_string[i: i + j]
            if not is_valid_part(part2):
                continue
            for k in range(1, 4):
                part3 = ip_string[i + j: i + j + k]
                if not is_valid_part(part3):
                    continue
                for l in range(1, 4):
                    part4 = ip_string[i + j + k: i + j + k + l]
                    if not is_valid_part(part4) or i + j + k + l != len(ip_string):
                        continue
                    valid_ip_address = part1 + "." + part2 + "." + part3 + "." + part4
                    valid_ip_addresses.append(valid_ip_address)

    return valid_ip_addresses


def add_roman(val, i):
    val + ()


def get_valid_ip_addresses_rec(ip_string, max_parts=5):
    def build_valid_ip_address(part_id, part_value):
        print(f" {part_id} ==> {part_value}")
        count[0] += 1
        if part_id >= max_parts:
            valid_ip_addresses.append(".".join(valid_ip_address))
        else:
            for i in range(1, 4):
                if len(part_value) < i or len(part_value) > ((max_parts - part_id) * 3):
                    continue
                if part_id < max_parts - 1:
                    part = part_value[0: i]
                    if is_valid_part(part):
                        valid_ip_address[part_id] = part
                        next_part = part_id + 1
                        next_part_len = len(part_value[i:])
                        min_len = max_parts - next_part
                        max_len = min_len * 3
                        if min_len <= next_part_len <= max_len:
                            build_valid_ip_address(next_part, part_value[i:])
                else:
                    if is_valid_part(part_value):
                        valid_ip_address[part_id] = part_value
                        build_valid_ip_address(part_id + 1, "END")
                    break

    valid_ip_address = [-1] * max_parts
    valid_ip_addresses = []
    count = [0]

    build_valid_ip_address(0, ip_string)
    print(f"Number of executions ==> {count[0]}")
    return valid_ip_addresses


# 6.11 Compute snake string
def snake_string(value):
    return value[1::4] + value[0::2] + value[3::4]


# 6.13 Match sub string rolling
def sub_string_present(text, pattern):
    looking_for_char_index = 0
    result = False
    for c in text:
        if c == pattern[looking_for_char_index]:
            if looking_for_char_index == len(pattern) - 1:
                result = True
                break
            else:
                looking_for_char_index += 1
        else:
            looking_for_char_index = 0
    return result


def sub_string_index(text, pattern):
    if len(text) < len(pattern):
        return -1

    BASE = 26
    text_hash = reduce(lambda carry, val: carry * BASE + ord(val), text[:len(pattern)], 0)
    pattern_hash = reduce(lambda carry, val: carry * BASE + ord(val), pattern, 0)

    power_s = BASE ** max((len(pattern) - 1), 0)

    for i in range(len(pattern), len(text)):
        if text_hash == pattern_hash and text[i - len(pattern): i] == pattern:
            return i - len(pattern)
        text_hash -= ord(text[i - len(pattern)]) * power_s
        text_hash = text_hash * BASE + ord(text[i])

    if text_hash == pattern_hash and text[-len(pattern):] == pattern:
        return len(text) - len(pattern)


if __name__ == "__main__":
    print(sub_string_index('AABCD', 'BCD'))

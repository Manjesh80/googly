import sys


def count_nume_of_bits(x):
    bit_count = 0;
    while x:
        bit_count += x & 1
        x >>= 1
    return bit_count


a = 16
b = 32


print(float('inf'))

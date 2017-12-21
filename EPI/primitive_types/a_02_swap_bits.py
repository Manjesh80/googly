def swap_bits(x, i, j):
    if i <= j:
        raise ValueError()
    if ((x >> i) & 1) != ((x >> j) & 1):
        mask = (1 << i) | (1 << j)
        x = x ^ mask

    return x


if __name__ == "__main__":
    swap_bits(int(0b1001001), 6, 1)

Mapping = ['0', '1', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ']


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


if __name__ == "__main__":
    number = '1234'
    final_res = None
    for i in reversed(range(0, len(number))):
        final_res = cross_the_list(mnemonic(int(number[i])), final_res)
    print(len(final_res))
    print(final_res)

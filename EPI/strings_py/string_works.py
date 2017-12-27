Mapping = ['0', '1', 'ABC', 'DEF', 'GHI']


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


if __name__ == "__main__":
    number = '1234'
    final_res = None
    for i in reversed(range(0, len(number))):
        final_res = cross_the_list(mnemonic(int(number[i])), final_res)
    print(len(final_res))
    print(final_res)

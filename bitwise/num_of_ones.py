# page 23
x = 8
no_of_bits_set_to_1 = {}
while x != 0:
    y = x & (~(x - 1))
    print(y)
    no_of_bits_set_to_1[y] = y
    x = x - y

print(len(no_of_bits_set_to_1))

# TODO Do for negative numbers which is two's complement

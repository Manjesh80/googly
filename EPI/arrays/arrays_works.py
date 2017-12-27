def separate_even_odd(arr):
    even_spot = 0
    odd_spot = len(arr) - 1
    while even_spot < odd_spot:
        if arr[even_spot] % 2 == 0:
            even_spot += 1
        else:
            arr[even_spot], arr[odd_spot] = arr[odd_spot], arr[even_spot]
            odd_spot -= 1


def multiply_two_arrays(m, n):
    res = [0] * (len(m) + len(n))
    for i in reversed(range(0, len(m))):
        carry = 0
        for j in reversed(range(0, len(n))):
            temp_sum = m[i] * n[j] + res[i + j + 1] + carry
            res[i + j + 1] = temp_sum % 10
            carry = temp_sum // 10
    res[0] = carry
    return res


if __name__ == "__main__":
    # arr = [*range(10)]
    # separate_even_odd(arr)
    res = multiply_two_arrays([1, 2, 1], [1, 2, 2])
    print(res)

def separate_even_odd(arr):
    even_spot = 0
    odd_spot = len(arr) - 1
    while even_spot < odd_spot:
        if arr[even_spot] % 2 == 0:
            even_spot += 1
        else:
            arr[even_spot], arr[odd_spot] = arr[odd_spot], arr[even_spot]
            odd_spot -= 1


# 5.1 - Dutch national flag ==> TODO

# 5.2 - Increment an arbitrary precision integer  ==> TODO

# 5.3 - Multiply two arrays
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


# 5.4 - Advancing through an array , whether you can jump and go to end ==> TODO

# 5.5 - Delete duplicates from sorted array ==> TODO

# 5.6 - Buy and sell stock once ==> TODO

# 5.7 - Buy and sell stock twice ==> TODO

# 5.8 - Computing and alternation a <= b >= c <= d >= e ==> TODO

# 5.9 - Enumerate all the primes of N  ==> TODO -> understand the formula
def generate_prime(n):
    size = ((n - 3) // 2) + 1
    primes = [2]
    is_prime = [True] * size
    for i in range(size):
        if is_prime[i]:
            p = 2 * i + 3
            primes.append(p)
            # Formula ( 2 * i ** 2 + 6 * i + 3 )
            for j in range((2 * i ** 2 + 6 * i + 3), size, p):
                is_prime[j] = False
    return primes


# 5.10 - Permute the elements of a Array ==> TODO ==> Do whiteboard and implement cyclic permutation sample
def permute_array_with_o1_space(A, perm):
    for i in range(0, len(A)):
        next = i
        while perm[next] >= 0:
            A[i], A[perm[next]] = A[perm[next]], A[i]
            temp = perm[next]
            perm[next] = perm[next] - len(perm)
            next = temp
    return A


# 5.11 - Compute the next permutation
def compute_next_permutation(A):
    inverse_point = len(A) - 2

    while inverse_point > 0 and A[inverse_point] > A[inverse_point + 1]:
        inverse_point -= 1

    if inverse_point == -1:
        return []

    # Replace inverse point with next highest value in the sub-chain
    for i in reversed(range(inverse_point, len(A))):
        if A[inverse_point] < A[i]:
            A[inverse_point], A[i] = A[i], A[inverse_point]
            break

    # Sort the list after inverse point and append
    A[inverse_point + 1:] = sorted(A[inverse_point + 1:])
    return A


# 5.12 - Sample offline data ==> TODO

# 5.13 - Sample online data ==> TODO

# 5.14 - Compute a Random permutation ==> TODO

# 5.15 - Compute a Random subset ==> TODO

# 5.16 - Generate non-uniform random numbers ==> TODO

# 5.17 - The Sudoku checker problem ==> TODO

# 5.18 - Compute the Spiral ordering of a 2D array ==> TODO

# 5.19 - Rotate 2D array ==> TODO

# 5.20 - Compute Rows in Pascal's triangle ==> TODO


if __name__ == "__main__":
    res = None
    # arr = [*range(10)]
    # separate_even_odd(arr)
    # 5.3 test
    # res = multiply_two_arrays([1, 2, 1], [1, 2, 2])
    # print(res)
    # res = permute_array_with_o1_space(['a', 'b', 'c', 'd'], [2, 0, 1, 3])

    res = compute_next_permutation([6, 2, 3, 5, 4, 1, 0])
    print(res)
    pass

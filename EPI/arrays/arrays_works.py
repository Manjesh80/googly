from random import randint
from collections import Counter
from itertools import islice
from random import randrange
import itertools
import random
import bisect


def separate_even_odd(arr):
    even_spot = 0
    odd_spot = len(arr) - 1
    while even_spot < odd_spot:
        if arr[even_spot] % 2 == 0:
            even_spot += 1
        else:
            arr[even_spot], arr[odd_spot] = arr[odd_spot], arr[even_spot]
            odd_spot -= 1


# 5.1 - Dutch national flag
# A = [2, 1, 0, 2, 1, 0, 2, 1, 0]
def dutch_national_flag(A, pivot_value):
    middle_head, middle_tail, greater_head = 0, 0, len(A) - 1
    current_index = 0
    middle_element_count = 0
    while middle_tail <= greater_head:
        if A[current_index] < pivot_value:
            A[current_index], A[middle_head] = A[middle_head], A[current_index]
            middle_head = middle_head + 1
            middle_tail = middle_head + middle_element_count
            current_index += 1
        elif A[current_index] == pivot_value:
            middle_element_count += 1
            middle_tail = middle_head + middle_element_count
            current_index += 1
        elif A[current_index] > pivot_value:
            A[current_index], A[greater_head] = A[greater_head], A[current_index]
            greater_head -= 1


def dutch_national_flag_epi(A, pivot_value):
    smaller, equal, larger = 0, 0, len(A)
    while equal < larger:
        if A[equal] < pivot_value:
            A[smaller], A[equal] = A[equal], A[smaller]
            smaller, equal = smaller + 1, equal + 1
        elif A[equal] == pivot_value:
            equal = equal + 1
        elif A[equal] > pivot_value:
            larger -= 1
            A[larger], A[equal] = A[equal], A[larger]


# 5.2 - Increment an arbitrary precision integer
def increment_integer(A):
    A[len(A) - 1] += 1
    for i in reversed(range(0, len(A))):
        if A[i] <= 9:
            break
        A[i], A[i - 1] = A[i] % 10, A[i - 1] + A[i] // 10

    if A[0] > 10:
        A.insert(0, 0)
        A[0], A[1] = A[1] // 10, A[1] % 10


# 5.3 Multiply arbitrary integer
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


# 5.4 - Advancing through an array , whether you can jump and go to end
# A = [3,3,1,0,2,0,1]
def advance_through_an_array(A):
    available_step = -1
    for v in A:
        available_step -= 1
        available_step = max(v, available_step)
        if available_step == 0:
            break
    return False if available_step == 0 else True


# 5.5 - Delete duplicates from sorted array ==> TODO
# A = [2,3,5,5,5,7,11,11,11,13]
def del_duplicate_from_sorted_array(A):
    write_idx = 0;
    for i in range(1, len(A)):
        if A[i] != A[i - 1]:
            write_idx += 1
            A[write_idx] = A[i]
    return A[0:write_idx + 1]


# 5.6 - Buy and sell stock once
def buy_and_sell_only_once(A):
    if A:
        max_profit = float('-inf')
        current_buy_rate = A[0]
        profit_today = [max_profit]
        current_buy_index = 0
        winning_pair = (0, 0)
        for current_idx in range(1, len(A)):

            current_profit = A[current_idx] - current_buy_rate
            profit_today.append(current_profit)
            if current_profit > max_profit:
                max_profit = current_profit
                winning_pair = (current_buy_index, current_idx)

            if A[current_idx] < A[current_buy_index]:
                current_buy_index = current_idx
                current_buy_rate = A[current_idx]
        return max_profit, winning_pair, profit_today
    else:
        return None


# 5.7 - Buy and sell stock twice
def buy_and_sell_only_twice(A):
    res = buy_and_sell_only_once(A)
    max_profit = 0
    if res:
        max_profit = res[0]
        left_trade, right_trade = A[0:res[1][0]], A[res[1][1] + 1:]
        left_res = buy_and_sell_only_once(left_trade)
        right_res = buy_and_sell_only_once(right_trade)
        if left_res and right_res:
            max_profit += max(left_res[0], right_res[0])
        elif left_res:
            max_profit += left_res[0]
        elif right_res:
            max_profit += right_res[0]

    return max_profit


def buy_and_sell_only_twice_epi(prices):
    max_total_profit, min_price_so_far = 0.0, float('inf')
    first_buy_sell_profits = [0] * len(prices)

    for i, price in enumerate(prices):
        min_price_so_far = min(min_price_so_far, price)
        max_total_profit = max(max_total_profit, price - min_price_so_far)
        first_buy_sell_profits[i] = max_total_profit

    print(first_buy_sell_profits)

    max_price_so_far = float('-inf')
    for i, price in reversed(list(enumerate(prices[1:], 1))):
        max_price_so_far = max(max_price_so_far, price)
        max_total_profit = max(max_total_profit,
                               max_total_profit - price + first_buy_sell_profits[i - 1])

    return max_total_profit


# 5.8 - Computing and alternation a <= b >= c <= d >= e
def rearrange(A):
    for i in range(len(A)):
        A[i:i + 2] = sorted(A[i:i + 2], reverse=i % 2)


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


# 5.10 - Permute the elements of a Array ==>
# TODO ==> Do whiteboard and implement cyclic permutation sample
def permute_array_with_o1_space(A, perm):
    for i in range(0, len(A)):
        next = i
        while perm[next] >= 0:
            A[i], A[perm[next]] = A[perm[next]], A[i]
            temp = perm[next]
            perm[next] = perm[next] - len(perm)
            next = temp
    return A


def permute_array_ganesh(A, perm):
    current_index = 0
    while current_index < len(perm) - 1:
        if perm[current_index] == current_index:
            current_index += 1
            continue
        item_getting_moved = A[perm[current_index]]
        A[current_index], A[perm[current_index]] = item_getting_moved, A[current_index]
        b = perm[current_index]
        perm[current_index], perm[b] = perm[b], perm[current_index]

    return A, perm


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


# 5.12 - Sample offline data
def generate_random_offline_data(A, k):
    for i in range(0, k):
        r = randint(0, len(A) - 1)
        A[i], A[r] = A[r], A[i]
    return A[0:k]


# 5.13 - Sample online data
def ordered_seq_generator(n=100):
    i = 1
    while i < n:
        yield i
        i += 1


def generate_random_sample_online(it, k):
    sampling_result = list(islice(it, k))
    numbers_seen_so_far = k

    for x in it:
        numbers_seen_so_far += 1
        idx_replace = randrange(numbers_seen_so_far)
        if idx_replace < k:
            sampling_result[idx_replace] = x
    return sampling_result


# 5.14 - Compute a Random permutation
def generate_random_permutation(A):
    for i in range(0, len(A)):
        r = randint(0, len(A) - 1)
        A[r], A[i] = A[i], A[r]
    return A


# 5.15 - Compute a Random subset. This is to get random subset from Combination of integers.
# brute force you can pick K number as random_int(0,n ), but you will get duplicates, so
# challenge is to avoid it
def random_subset(n, k):
    pass


# 5.16 - Generate non-uniform random numbers
def generate_non_uniform_random_numbers(values, probability):
    prefix_sum_of_prob = ([0.0] + list(itertools.accumulate(probability)))
    rand_value = random.random()
    interval_idx = bisect.bisect(prefix_sum_of_prob, rand_value) - 1
    return values[interval_idx]


# 5.17 - The Sudoku checker problem ==> TODO

# 5.18 - Compute the Spiral ordering of a 2D array ==> TODO
# A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# B = matrix_in_spiral_order(A)
def matrix_in_spiral_order(square_matrix):
    SHIFT = ((0, 1), (1, 0), (0, -1), (-1, 0))
    direction = x = y = 0
    spiral_ordering = []

    for _ in range(len(square_matrix) ** 2):
        spiral_ordering.append(square_matrix[x][y])
        square_matrix[x][y] = 0
        next_x, next_y = x + SHIFT[direction][0], y + SHIFT[direction][1]
        if (next_x not in range(len(square_matrix)) or
                next_y not in range(len(square_matrix)) or
                square_matrix[next_x][next_y] == 0):
            direction = (direction + 1) & 3
            next_x, next_y = x + SHIFT[direction][0], y + SHIFT[direction][1]
        x, y = next_x, next_y

    return spiral_ordering


# 5.19 - Rotate 2D array ==> TODO


# 5.20 - Compute Rows in Pascal's triangle ==> TODO


if __name__ == "__main__":
    res = permute_array_with_o1_space(['a', 'b', 'c', 'd'], [2, 0, 1, 3])
    print(res)
    # for i in range(100, 105):
    #     print(i, '--> ', ~i)
    # print('*****')
    # for i in range(5):
    #     print(i, '--> ', ~i)

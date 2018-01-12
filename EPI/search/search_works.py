import math
import random
import operator


# 11.1 Find first occurrence of where array has duplicates
def find_first_occurrence_k_has_duplicates(sorted_array, k):
    result_index = -1
    left, right = 0, len(sorted_array)
    while left < right:
        mid = left + (right - left) // 2
        mid_value = sorted_array[mid]
        if k < mid_value:
            right = mid - 1
        elif k == mid_value:
            result_index = mid
            right = mid - 1
        else:
            left = mid + 1
    return result_index


# 11.2 Search a sorted array to see if any matching index
def search_a_sorted_array_for_entry_equal_to_index(sorted_array):
    result_index = -1
    left, right = 0, len(sorted_array)
    while left < right:
        mid_index = (left + right) // 2
        mid_value = sorted_array[mid_index]

        if mid_index > mid_value:
            left = mid_index + 1
        elif mid_value == mid_index:
            result_index = mid_index
            break
        else:
            right = mid_index - 1

    return result_index


# 11.3 Search a cyclically sorted array
# This is bit of crap logic, so thinking a simple is alawys better
def search_starting_point_of_cyclic_sorted_array(sorted_array):
    result_index = -1
    left, right = 0, len(sorted_array)
    while left < right:
        mid_index = (left + right) // 2
        left_mid_index = (left + mid_index) // 2
        right_mid_index = (mid_index + 1 + right) // 2

        if sorted_array[mid_index] < sorted_array[left_mid_index]:
            right = mid_index - 1
        elif sorted_array[mid_index] < sorted_array[right_mid_index]:
            left = mid_index + 1
        elif left_mid_index == right_mid_index - 1:
            left = left_mid_index if sorted_array[left_mid_index] < sorted_array[
                right_mid_index] else right_mid_index
            break

    return left


def search_starting_point_of_cyclic_sorted_array_epi(sorted_array):
    left, right = 0, len(sorted_array) - 1
    while left < right:
        mid = (left + right) // 2
        if sorted_array[mid] > sorted_array[right]:
            left = mid + 1
        else:
            right = mid
    return left


# 11.4 Compute nearest integer square root
# Think better and write the algorithm
def compute_nearest_integer_square_root(num):
    left, right = 0, num
    while left <= right:
        mid = (left + right) // 2
        mid_sqr = mid * mid
        if mid_sqr <= num:
            left = mid + 1
        else:
            right = mid - 1

    return left - 1


# 11.5 Calculate real square root
def calculate_real_root(num):
    left, right = (1.0, num)

    while not math.isclose(left, right):
        mid = (left + right) / 2
        mid_sqr = mid * mid
        if num > mid_sqr:
            left = mid
        else:
            right = mid

    return left


# 11.8 Find the k-th largest with O(n) best case O(n**2) worst case,
# we can achieve a static of (n log k) with space of O( log k)
def find_kth_largest(k, A, B):
    def find_kth(comp):
        def partition_around_pivot(left, right, pivot_idx):
            pivot_value = A[pivot_idx]
            new_pivot_idx = left
            l[0] += 1
            print(f"********* Iteration {l[0] } *********")
            print(f" Original input ==> {B} ")
            print(f" Modified input ==> {A} ")
            print(f" Pivot Index = {pivot_idx} ==> Pivot Value => {pivot_value} ")

            A[pivot_idx], A[right] = A[right], A[pivot_idx]
            for i in range(left, right):
                if comp(A[i], pivot_value):
                    A[i], A[new_pivot_idx] = A[new_pivot_idx], A[i]
                    new_pivot_idx += 1
            A[right], A[new_pivot_idx] = A[new_pivot_idx], A[right]
            print(f"Array changed ==> {A}")
            print(f"********* Iteration {l[0] } END *********")
            return new_pivot_idx

        left, right = 0, len(A) - 1
        while left <= right:
            pivot_idx = random.randint(left, right)
            new_pivot_idx = partition_around_pivot(left, right, pivot_idx)
            print('************')
            print(A)
            print('************')
            if new_pivot_idx == k - 1:
                return A[new_pivot_idx]
            elif new_pivot_idx > k - 1:
                right = new_pivot_idx - 1
            else:
                left = new_pivot_idx + 1

    l = [0]
    return find_kth(operator.gt)


if __name__ == "__main__":
    input = random.sample(range(1, 11), 10)
    # input = [10, 9, 8, 2, 5, 4, 1, 6, 7, 3]
    print('**********************')
    print(input)
    print('**********************')
    print(find_kth_largest(4, input, input))
    print('**********************')

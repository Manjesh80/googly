import math


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


if __name__ == "__main__":
    print(calculate_real_root(99))

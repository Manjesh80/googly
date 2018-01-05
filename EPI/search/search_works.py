# 10.1 Find first occurrence of where array has duplicates
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


# 10.2 Search a sorted array to see if any matching index
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


if __name__ == "__main__":
    sorted_array = [-2, 0, 2, 3, 2, 2, 3]
    print(search_a_sorted_array_for_entry_equal_to_index(sorted_array))

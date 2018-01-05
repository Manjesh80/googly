from heapq import *
from itertools import *


# 10.1 Merge sorted array
def merge_sorted_array(sorted_arrays):
    arr_iter_dict = {i: (i, iter(a)) for i, a in enumerate(sorted_arrays)}
    sorting_list = []
    for key in arr_iter_dict.keys():
        i, itr = arr_iter_dict[key]
        val = next(itr, None)
        if val:
            heappush(sorting_list, (val, i))

    sorted_list = []
    while sorting_list:
        min_value = heappop(sorting_list)
        sorted_list.append(min_value[0])
        next_idx, next_itr = arr_iter_dict[min_value[1]]
        val = next(next_itr, None)
        if val:
            heappush(sorting_list, (val, next_idx))

    return sorted_list


# 10.2 Sort K ascending and descending
def sort_k_increasing_decresing_array(arr):
    current_run_elements = 0
    sorted_arrays = []
    # for i in range(1, len(arr) + 1):
    i = 1
    while i <= len(arr):
        # If 2nd element greater the good .. do
        if arr[i] >= arr[i - 1]:
            current_run_elements += 1
            i += 1
            continue
        # If 2nd element smaller the ,create a subset and move on
        elif arr[i] < arr[i - 1] and current_run_elements == 0:
            sorted_arrays.append([arr[i], arr[i - 1]])
            current_run_elements = 0
            i += 2
        # If 2nd element smaller the ,create a subset and move on
        elif arr[i] < arr[i - 1] and current_run_elements > 0:
            sorted_arrays.append(arr[i - current_run_elements - 1:i])
            current_run_elements = 0
            i += 1

    return sorted_arrays


# 10.3 Sort an almost sorted array
def sort_approximately_sorted_array(sequence, k):
    min_heap = []
    for x in islice(sequence, k):
        heappush(min_heap, x)

    for x in sequence:
        smallest = heappushpop(min_heap, x)
        print(smallest)

    while min_heap:
        smallest = heappop(min_heap)
        print(smallest)


# 10.5 Compute online median
def compute_online_median(sequence):
    leftie, rightie = [], []
    median_seq = []
    left_maxie = float('inf')
    right_minie = float('inf')
    for i in range(1, len(sequence) + 1):
        v = sequence[i - 1]
        if i % 2 == 1:
            if v < right_minie:
                heappush(leftie, -v)
            else:
                heappush(leftie, -heappop(rightie))
                heappush(rightie, v)
        else:
            if v > left_maxie:
                heappush(rightie, v)
            else:
                heappush(rightie, -heappop(leftie))
                heappush(leftie, -v)

        left_maxie = -leftie[0]
        right_minie = rightie[0] if len(rightie) > 0 else 0
        if i % 2 == 1:
            median_seq.append(-leftie[0])
        else:
            median_seq.append((-leftie[0] + rightie[0]) / 2)

    return median_seq


def compute_online_median_epi(sequence):
    min_heap = []  # max_heap
    max_heap = []  # min_heap

    for x in sequence:
        heappush(max_heap, -heappushpop(min_heap, x))
        if len(max_heap) > len(min_heap):
            heappush(min_heap, -heappop(max_heap))

        print(0.5 * (min_heap[0] + (-max_heap[0]))
              if len(min_heap) == len(max_heap) else min_heap[0])


if __name__ == "__main__":
    # merge_sorted_array([[2, 20, 200], [3, 30, 300], [4, 40, 400]])
    l = [1, 0, 3, 5, 2, 0, 1]
    print(compute_online_median_epi(l))

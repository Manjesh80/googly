from collections import Counter
from collections import namedtuple


class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name

    def __gt__(self, other):
        return self.name > other.name

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return "( " + self.name + " , " + str(self.age) + " )"

    def __repr__(self):
        return "( " + self.name + " , " + str(self.age) + " )"


def simple_list_sorted_by_name():
    students = [Student('D', 5), Student('A', 30), Student('A', 35), Student('B', 20), Student('C', 10)]
    new_students = sorted(students)
    print(students)
    print(new_students)
    print(sorted(students, key=lambda s: (s.name, s.age), reverse=True))
    print(students)


# 13.1 Compute the intersection of the sorted array
def compute_intersection_of_sorted_array(A, B):
    # A, B = A, B if (len(A) <= len(B)) else B, A
    if len(A) > len(B):
        A, B = B, A
    a_i = 0
    b_i = 0
    results = []
    while a_i < len(A) and b_i < len(B):
        while a_i < len(A) - 1 and A[a_i] == A[a_i + 1]:
            a_i += 1
        while b_i < len(B) - 1 and B[b_i] == B[b_i + 1]:
            b_i += 1

        if A[a_i] < B[b_i]:
            a_i += 1
            pass
        elif A[a_i] == B[b_i]:
            results.append(A[a_i])
            a_i, b_i = a_i + 1, b_i + 1
        else:
            b_i += 1

    return results


# 13.2 merge_two_sorted_arrays_with_blank_space
def merge_two_sorted_arrays_with_blank_space(A, B):
    a_i = len(A) - 1
    b_i = len(B) - 1
    a_w_i = len(A) - 1
    while A[a_i] is None:
        a_i -= 1

    while a_i > -1 and b_i > -1:
        if A[a_i] > B[b_i]:
            A[a_w_i] = A[a_i]
            a_i -= 1
            a_w_i -= 1
        else:
            A[a_w_i] = B[b_i]
            b_i -= 1
            a_w_i -= 1

    while b_i >= 0:
        A[a_w_i] = B[b_i]
        b_i -= 1
        a_w_i -= 1
    return A


# 13.3 Remove first name duplicates
def remove_first_name_duplicates(names):
    names.sort()
    write_idx = 1
    for student in names[1:]:
        if student != names[write_idx - 1]:
            names[write_idx] = student
            write_idx += 1
    del names[write_idx:]


# 13.4 Smallest non constructable value
def small_non_constructable_value(A):
    A.sort()
    min_non_constructable_value = 0
    for i in range(0, len(A)):
        if A[i] > min_non_constructable_value + 1:
            break
        min_non_constructable_value += A[i]
    return min_non_constructable_value + 1


# 13.5 Number of simultaneous events
def num_of_concurrent_interviews(meetings):
    Event_Points = namedtuple("Event_Points", ("time", 'is_start'))

    event_coordinates = []
    for meeting in meetings:
        event_coordinates += [Event_Points(meeting[0], True),
                              Event_Points(meeting[1], False)]

    event_coordinates.sort(key=lambda m: (m.time, m.is_start))

    max_concurr_events, concurr_events = 0, 0
    for event_coordinate in event_coordinates:
        if event_coordinate.is_start:
            concurr_events += 1
            max_concurr_events = max(max_concurr_events, concurr_events)
        else:
            concurr_events -= 1

    return max_concurr_events


# 13.6 Merge interval
def merge_interval(schedule, new_meet):
    def does_new_meeting_start_overlap(meeting):
        return new_meet[0] >= meeting[0]

    def does_new_meeting_end_overlap(meeting):
        return new_meet[1] <= meeting[1]

    def does_new_meeting_completely_overlap(meeting):
        return does_new_meeting_start_overlap(meeting) and \
               does_new_meeting_end_overlap(meeting)

    start_overlap_index, end_overlap_index = -1, -1
    for idx, meeting in enumerate(schedule):
        if does_new_meeting_completely_overlap(meeting):
            return
        elif does_new_meeting_start_overlap(meeting):
            start_overlap_index = idx
        elif does_new_meeting_end_overlap(meeting) and end_overlap_index == -1:
            end_overlap_index = idx

    if start_overlap_index != -1:
        new__merged_meet = (schedule[start_overlap_index][0], schedule[end_overlap_index][1])
        del schedule[start_overlap_index:end_overlap_index + 1]
        schedule.insert(start_overlap_index, new__merged_meet)

    return schedule


# heap sort
def heapify_custom(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root
    if l < n and arr[i] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify_custom(arr, n, largest)


# The main function to sort an array of given size
def heapSort(arr):
    n = len(arr)
    for i in range(n, -1, -1):
        heapify_custom(arr, n, i)


def heapify_min(A, len_of_array, tail_index):
    current_tail_index = tail_index
    left_index = 2 * tail_index + 1
    right_index = 2 * tail_index + 2

    if right_index < len_of_array and A[right_index] < A[tail_index]:
        current_tail_index = right_index

    if left_index < len_of_array and A[left_index] < A[tail_index]:
        current_tail_index = left_index

    if tail_index != current_tail_index:
        if A[current_tail_index] < A[tail_index]:
            A[tail_index], A[current_tail_index] = A[current_tail_index], A[tail_index]
        heapify_min(A, len_of_array, current_tail_index)


def heap_min(A):
    len_of_array = len(A)
    for i in range(len_of_array, -1, -1):
        heapify_min(A, len_of_array, i)


def heap_pop(A):
    heap_min(A)
    res = A[0]
    del A[0]
    return res


# 13.7 Compute union of intervals
def union_interval(intervals):
    intervals.sort(key=lambda k: k[0])
    results = []

    new_session = True
    current_run_start_idx = -1
    current_run_end = -1
    for idx, interval in enumerate(intervals):
        if new_session:
            current_run_start_idx = idx
            current_run_end = interval[1]
            new_session = False
        else:
            if interval[0] > current_run_end or interval[1] == current_run_end:
                results.append((current_run_start_idx, idx))
                new_session = True
                continue
            else:
                current_run_end = max(current_run_end, interval[1])

    results.append((current_run_start_idx, idx))
    return results


# 13.8 sorted_keys_together
def sort_key_together(students):
    id_counter = Counter([student[0] for student in students])
    key_with_idx = {}
    current_start_index = 0
    for student_id, student_count in id_counter.items():
        key_with_idx[student_id] = (current_start_index, (current_start_index + student_count - 1))

    for student in students:
        student_id_start_index, student_id_end_index = key_with_idx[student_id]
    return None


Person = namedtuple('Person', ('age', 'name'))


# 13.8 sorted_keys_together
def sort_key_together_epi(people):
    age_to_count = Counter([p.age for p in people])
    age_to_offset, offset = {}, 0
    for age, count in age_to_count.items():
        age_to_offset[age] = offset
        offset += count

    while age_to_offset:
        from_age = next(iter(age_to_offset))
        from_idx = age_to_offset[from_age]
        to_age = people[from_idx].age
        to_idx = age_to_offset[to_age]
        people[from_idx], people[to_idx] = people[to_idx], people[from_idx]
        age_to_count[to_age] -= 1
        if age_to_count[to_age]:
            age_to_offset[to_age] = to_idx + 1
        else:
            del age_to_offset[to_age]


# 13.9 Team photo day 1
def team_photo_day_one(team1, team2):
    team1.sort()
    team2.sort()
    if team1[0] < team2[0]:
        return all(t1 < t2 for t1, t2 in zip(team1, team2))
    else:
        return all(t1 > t2 for t1, t2 in zip(team1, team2))


class ListNode:
    def __init__(self, data=0, next_node=None):
        self.data = data
        self.next_node = next_node


# 7.1 merge two sorted list
def merge_two_sorted_list(L1, L2):
    dummy_head = tail = ListNode()
    while L1 and L2:
        if L1.data < L2.data:
            tail.next_node = L1
            L1 = L1.next_node
        else:
            tail.next_node = L2
            L2 = L2.next_node
        tail = tail.next_node

    # Append remaining
    tail.next_node = L1 or L2
    return dummy_head.next_node


# 13.10 Stable sort list
def stable_sort_list(L):
    if not L or not L.next_node:
        return L

    pre_slow, slow, fast = None, L, L
    while fast and fast.next_node:
        pre_slow = slow
        slow, fast = slow.next_node, fast.next_node.next_node

    if pre_slow:
        pre_slow.next_node = None
    return merge_two_sorted_list(stable_sort_list(L), stable_sort_list(slow))


# 13.11 Compute Salary threshold
def find_salary_cap(target_payroll, current_salaries):
    current_salaries.sort()
    unadjusted_salary_sum = 0.0
    for i, current_salary in enumerate(current_salaries):
        adjusted_people = len(current_salaries) - i
        adjusted_sal = current_salary * adjusted_people
        if (unadjusted_salary_sum + adjusted_sal) >= target_payroll:
            return (target_payroll - unadjusted_salary_sum) / adjusted_people
        unadjusted_salary_sum += current_salary
    return -1


if __name__ == "__main__":
    res = find_salary_cap(210, [20, 30, 40, 90, 100])
    print(res)

    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top
    # Comment to move to top

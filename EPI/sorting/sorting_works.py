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


if __name__ == "__main__":
    meetings = [(1, 4), (2, 4), (3, 4)]
    res = num_of_concurrent_interviews(meetings)
    print(res)

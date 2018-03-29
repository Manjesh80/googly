from collections import namedtuple
from bintrees import RBTree


# 24.5 Longest contiguous sub-array
def longest_contiguous_sub_array(A):
    if not A:
        return None
    prev = A[0]
    current_run_length, curr_start_index, curr_end_index = 1, 0, 0
    best_run_length, best_start_index, best_end_index = 1, 0, 0
    for i in range(1, len(A) - 1):
        curr = A[i]
        if curr > prev:
            current_run_length += 1
        else:
            if current_run_length > best_run_length:
                best_run_length, best_start_index, best_end_index = current_run_length, curr_start_index, i - 1
            current_run_length, curr_start_index, curr_end_index = 1, i, i
        prev = curr

    return (best_run_length, best_start_index, best_end_index)


# 24.25 View from above
LineSegement = namedtuple('LineSegement', ('left', 'right', 'color', 'height'))


def calculate_view_from_above(A):
    class Endpoint:
        def __init__(self, is_left, line):
            self.is_left = is_left
            self.line = line

        def __lt__(self, other):
            return self.value() < other.value()

        def value(self):
            return self.line.left if self.is_left else self.line.right

    sorted_endpoints = sorted([Endpoint(True, a) for a in A] +
                              [Endpoint(False, a) for a in A])
    result = []
    prev_xaxis = sorted_endpoints[0].value()
    prev = None
    active_line_segments = RBTree()
    for endpoint in sorted_endpoints:
        if active_line_segments and prev_xaxis != endpoint.value():
            active_segment = active_line_segments.max_item()[1]
            if prev is None:
                prev = LineSegement(prev_xaxis, endpoint.value(),
                                    active_segment.color,
                                    active_segment.height)
            else:
                if (prev.height == active_segment.height and
                        prev.color == active_segment.color and
                        prev_xaxis == prev.right):
                    prev = prev._replace(right=endpoint.value())
                else:
                    result.append(prev)
                    prev = LineSegement(prev_xaxis, endpoint.value(),
                                        active_segment.color, active_segment.height)

        prev_xaxis = endpoint.value()

        if endpoint.is_left:
            active_line_segments[endpoint.line.height] = endpoint.line
        else:
            del active_line_segments[endpoint.line.height]

    return result + [prev] if prev else result


Line = namedtuple('Line', ('x1', 'y1', 'x2', 'y2'))


class LineDetail:
    def __init__(self, line, is_start, is_perpendicular):
        self.line = line
        self.is_start = is_start
        self.is_perpendicular = is_perpendicular

    def __lt__(self, other):
        return self.line.x1 < other.line.x1


def identify_intersection_lines(L):
    sorted_line_details = sorted([LineDetail(l, True, l.x1 == l.x2) for l in L]) + \
                          sorted([LineDetail(l, False, l.x1 == l.x2) for l in L])
    sorted_tree = RBTree()
    result = []
    for line_detail in sorted_line_details:
        if not line_detail.is_start and not line_detail.is_perpendicular:
            del sorted_tree[line_detail.line.y1]
        elif line_detail.is_start and not line_detail.is_perpendicular:
            sorted_tree[line_detail.line.y1] = line_detail
        else:
            start, end = line_detail.line.y1, line_detail.line.y2
            start, end = (start, end) if start < end else (end, start)
            intersects = [*sorted_tree.key_slice(start, end)]
            result.append(((line_detail.line.x1, line_detail.line.y1), intersects.copy()))
    return result


# 24.32 Determine critical height
def determine_critical_height(cases, drops):
    def get_height_helper(cases, drops):
        if cases == 0 or drops == 0:
            return 0
        if cases == 1:
            return drops
        if F[cases][drops] == -1:
            F[cases][drops] = (get_height_helper(cases, drops - 1) +
                               get_height_helper(cases - 1, drops - 1) + 1)
        return F[cases][drops]

    F = [[-1] * (drops + 1) for _ in range(cases + 1)]
    res = get_height_helper(cases, drops)
    print(F)
    return res


if __name__ == "__main__":
    res = determine_critical_height(4, 5)
    print(res)
    # L = [Line(2, 2, 9, 2), Line(3, 5, 9, 25), Line(5, 7, 9, 7), Line(8, 1, 8, 9)]
    # res = identify_intersection_lines(L)
    # print(res)

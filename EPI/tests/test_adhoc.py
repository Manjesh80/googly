import bisect


def convert(s, numRows):
    result = [[] for _ in range(0, numRows)]
    i = 0
    char_count = 0
    move_down = True
    while char_count < len(s):
        result[i].append(s[char_count])
        char_count += 1
        if move_down:
            i += 1
        else:
            i -= 1

        if i == numRows - 1 or i == 0:
            move_down = not move_down

    final_str = ''
    for r in result:
        final_str += ''.join(r)

    return final_str


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def isValidBST(root):
    def pre_order(node):
        if not node:
            return True

        left_value = node.left.val if node.left else float('-inf')
        right_value = node.right.val if node.right else float('inf')
        node_res = (left_value < node.val < right_value)
        if not node_res:
            return node_res
        left_res = pre_order(node.left)
        if not left_res:
            return left_res
        right_res = pre_order(node.right)
        return node_res and left_res and right_res

    return pre_order(root)


def repeatedStringMatch_old(A, B):
    if len(A) > len(B):
        if A.find(B) >= 0:
            return 1
        else:
            return -1

    first_char = A[0]
    last_char = A[-1]
    first_char_first_idx = B.find(first_char)
    last_char_last_idx = B.rfind(last_char)
    num_of_good_A_in_B = (last_char_last_idx - first_char_first_idx + 1) // len(A)

    if first_char_first_idx != 0:
        num_of_good_A_in_B += 1

    if last_char_last_idx != (len(B) - 1):
        num_of_good_A_in_B += 1

    return num_of_good_A_in_B


def repeatedStringMatchold2(A, B):
    la = len(A)
    lb = len(B)
    if la > lb:
        if A.find(B) >= 0:
            return 1
        elif (A + A).find(B) >= 0:
            return 2
        else:
            return -1

    first_idx = B.find(A)
    if first_idx == -1:
        return first_idx
    elif first_idx > 0 and not A.endswith(B[0:first_idx]):  # bad first half
        return -1

    size_of_last_a = (lb - first_idx) % la
    num_of_good_A_in_B = (lb - first_idx) // la

    if first_idx != 0:
        num_of_good_A_in_B += 1

    if size_of_last_a > 0:
        if not A.startswith(B[-size_of_last_a:]):
            return -1
        else:
            num_of_good_A_in_B += 1

    return num_of_good_A_in_B


def repeatedStringMatch(A, B):
    la = len(A)
    lb = len(B)
    if la > lb:
        if A.find(B) >= 0:
            return 1
        elif (A + A).find(B) >= 0:
            return 2
        else:
            return -1

    first_idx = B.find(A)
    if first_idx == -1:
        if (A + A).find(B) >= 0:
            return 2
        else:
            return -1

    first_broken_A = B[0:first_idx]
    number_A_needed = 0
    if len(first_broken_A) >= len(A) or not A.endswith(first_broken_A):
        return -1
    if len(first_broken_A) != 0:
        number_A_needed += 1

    size_of_last_a = (lb - first_idx) % la
    last_broken_A = B[-size_of_last_a:]
    if size_of_last_a > 0:
        if not A.startswith(last_broken_A):
            return -1
        else:
            number_A_needed += 1

    start_index = first_idx
    bad_word = False
    while start_index + len(A) <= lb:
        current_word = B[start_index:start_index + len(A)]
        if current_word != A:
            bad_word = True
            break
        number_A_needed += 1
        start_index += len(A)

    if bad_word:
        return -1

    return number_A_needed


def kEmptySlots(flowers, k):
    """
    :type flowers: List[int]
    :type k: int
    :rtype: int
    """
    number_of_blooming_flower = 0
    for flower in flowers:
        number_of_blooming_flower += 1
        if number_of_blooming_flower == 2:
            flower_value_diff = abs(flowers[1] - flowers[0])
            if flower_value_diff in flowers[2:]:
                return flower_value_diff
            else:
                return -1


class NewSolution():
    def nextClosestTime(self, time):
        available_digits = sorted([int(i) for i in list(time.replace(":", ""))])
        time_digits = [int(i) for i in list(time.replace(":", ""))]
        time_digits.reverse()
        limits = {1: 9, 2: 5, 3: [9, 3], 4: 2}
        limits_reverse = {4: 9, 3: 5, 2: [9, 3], 1: 2}
        pos_found, flip_pos, new_time_digits = self.flip_position(time_digits, available_digits, limits)
        if pos_found:
            for idx in range(0, flip_pos):
                self.fill_minimum(idx, new_time_digits, available_digits, limits)
            new_time_digits.reverse()
        else:
            new_time_digits = self.generate_minimum(available_digits, limits_reverse)

        new_time_digits.insert(2, ":")
        result = ''.join([str(i) for i in new_time_digits])
        return result

    def get_permutations(self, A, K):
        def directed_perm(pos, curr):
            if len(curr) == K:
                result.add(int("".join([str(i) for i in curr])))
                return
            for p in range(pos, len(A)):
                directed_perm(pos + 1, curr + A[p])

        result = set()
        return result

    def generate_minimum(self, available_digits, limits_reverse):
        min_time = []
        for idx in range(0, 4):
            min_value = float('-inf')
            max_value = limits_reverse[idx + 1]
            if idx == 1:
                max_value = limits_reverse[idx + 1][1] if min_time[0] == 2 else limits_reverse[idx + 1][0]
            found, result = self.get_next_value(available_digits, min_value, max_value)
            if found:
                min_time.append(result)
            else:
                raise ValueError('Next minimum failed')
        return min_time

    def fill_minimum(self, idx, new_time_digits, available_digits, limits):
        min_value = float('-inf')
        max_value = limits[idx + 1]
        if idx == 2:
            max_value = limits[idx + 1][1] if new_time_digits[3] == 2 else limits[idx + 1][0]
        found, result = self.get_next_value(available_digits, min_value, max_value)
        if found:
            new_time_digits[idx] = result
        else:
            raise ValueError('No minimum')

    def flip_position(self, time_digits, available_digits, limits):
        pos_found = False
        pos = -1
        new_time_digits = None
        for idx, curr_digit in enumerate(time_digits):
            min_value = curr_digit
            max_value = limits[idx + 1]
            if idx == 2:
                max_value = limits[idx + 1][1] if time_digits[3] == 2 else limits[idx + 1][0]
            found, result = self.get_next_value(available_digits, min_value, max_value)
            if found:
                new_time_digits = list(time_digits.copy())
                new_time_digits[idx] = result
                pos = idx
                pos_found = True
                break
        return pos_found, pos, new_time_digits

    def get_next_value(self, available_digits, min_value, max_value):
        found = False
        result = None
        for val in available_digits:
            if min_value < val <= max_value:
                found = True
                result = val
                break
        return found, result


class NewSolution():
    def nextClosestTime(self, time):
        return self.next_time(time)

    def next_time(self, time):
        hr, mm = int(time.split(":")[0]), int(time.split(":")[1])
        l = [int(i) for i in list(time.replace(":", ""))]
        vals = self.get_permutations((l + l), 2)

        next_min = self.next_high(vals, mm)
        if next_min and next_min < 60:
            return '{0:02d}'.format(hr) + ":" + '{0:02d}'.format(next_min)

        next_hr = self.next_high(vals, hr)
        if next_hr and next_hr < 24:
            return '{0:02d}'.format(next_hr) + ":" + '{0:02d}'.format(vals[0])

        return '{0:02d}'.format(vals[0]) + ":" + '{0:02d}'.format(vals[0])

    def get_permutations(self, A, K):
        def directed_perm(pos, curr):
            if len(curr) == K:
                result.add(int("".join([str(i) for i in curr])))
                return
            for p in range(pos, len(A)):
                directed_perm(p + 1, curr + [A[p]])

        result = set()
        directed_perm(0, [])
        return sorted(list(result))

    def next_high(self, A, V):
        low = 0
        high = len(A) - 1
        res = None
        while low <= high:
            mid = (low + high) // 2
            if A[mid] == V:
                res = None if (mid + 1) >= len(A) else A[mid + 1]
                break
            elif A[mid] > V:
                res = A[mid]
                high = mid - 1
            else:
                low = mid + 1
        return res


def licenseKeyFormatting(S, K):
    S = S.replace('-', '')
    S = S.upper()
    first_pos = (len(S) % K)
    first_chunk = (S[0:first_pos] + '-') if first_pos > 0 else ''
    rest = S[first_pos:]
    result = [rest[i:i + K] for i in range(0, len(rest), K)]
    return first_chunk + '-'.join(result)


class KSolution:
    def kEmptySlots(self, flowers, k):
        """
        :type flowers: List[int]
        :type k: int
        :rtype: int
        """
        import heapq
        min_heap = []
        max_heap = []
        slot_available = [True for _ in flowers]

        for idx, flower in enumerate(flowers):
            flower -= 1
            heapq.heappush(min_heap, flower)
            curr_slot = idx
            slot_available[flower] = False

            while min_heap and (min_heap[0] <= curr_slot):
                popped_element = heapq.heappop(min_heap)
                heapq.heappush(max_heap, -popped_element)

            if min_heap and max_heap:
                next_bloom_flower = min_heap[0]
                prev_bloom_flower = -max_heap[0]
                diff_slots = slot_available[prev_bloom_flower + 1:next_bloom_flower]
                if (len(diff_slots) == k) and all(diff_slots):
                    return idx + 1
        return -1


class KSolution2:
    def kEmptySlots(self, flowers, k):
        active = []
        for day, flower in enumerate(flowers, 1):
            i = bisect.bisect(active, flower)
            for neighbor in active[i - (i > 0):i + 1]:
                if abs(neighbor - flower) - 1 == k:
                    return day
            active.insert(i, flower)
        return -1

    def kEmptySlots_1(self, flowers, k):
        """
        :type flowers: List[int]
        :type k: int
        :rtype: int
        """
        import heapq
        min_heap = []
        heapq.heapify(min_heap)
        for idx, flower in enumerate(flowers):
            curr_day = idx + 1
            print(f"Processing day ==> {curr_day}")
            heapq.heappush(min_heap, flower)
            min_heap.sort()
            if curr_day in min_heap:
                next_blossom_day = curr_day + k + 1
                if next_blossom_day in min_heap:
                    if abs(min_heap.index(curr_day) - min_heap.index(next_blossom_day)) == 1:
                        return curr_day
            else:
                prev_blossom_day = curr_day - 1
                next_blossom_day = curr_day + k
                if prev_blossom_day in min_heap and next_blossom_day in min_heap:
                    if abs(min_heap.index(prev_blossom_day) - min_heap.index(next_blossom_day)) == 1:
                        return curr_day
            # from flower_index
            next_blossom_day_by_flower = flower + k + 1
            if next_blossom_day_by_flower in min_heap:
                if abs(min_heap.index(flower) - min_heap.index(next_blossom_day_by_flower)) == 1:
                    return curr_day
            prev_blossom_day_by_flower = flower - k - 1
            if prev_blossom_day_by_flower in min_heap:
                if abs(min_heap.index(flower) - min_heap.index(prev_blossom_day_by_flower)) == 1:
                    return curr_day
        return -1


class KSolution3:
    def kEmptySlots(self, flowers, k):
        active = []
        for day, flower in enumerate(flowers, 1):
            i = bisect.bisect(active, flower)
            for neighbor in active[i - (i > 0):i + 1]:
                if abs(neighbor - flower) - 1 == k:
                    return day
            active.insert(i, flower)
        return -1


class TSolution:
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        import collections
        bad_tree = False
        d = collections.defaultdict(list)
        for edge in edges:

            parent_node = d[edge[0]]
            child_node = d[edge[1]]

            if len(parent_node) == 0:
                parent_node = [0, 0]
            if len(child_node) == 0:
                child_node = [0, 0]

            parent_node[1] += 1
            if parent_node[1] >= 3:
                bad_tree = True
                break

            child_node[0] += 1
            if child_node[0] >= 2:
                bad_tree = True
                break

            d[edge[0]] = parent_node
            d[edge[1]] = child_node

        if (len(d) == n) and not bad_tree:
            return True
        else:
            return False


class GSolution:
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        if n == 0:
            return True

        if n == 1 and len(edges) == 0:
            return True
        if n > 1 and len(edges) == 0:
            return False

        import collections
        DS = collections.namedtuple('DS', ('anchor', 'data'))
        points_set = {}

        def find_anchor(point):
            if point in points_set:
                return point
            for anchor, data_set in points_set.items():
                if point in data_set.data:
                    return anchor
            return None

        def merge_data_set(anchor, data_set_1, data_set_2):
            return DS(anchor=anchor, data=data_set_1.union(data_set_2))

        bad_tree = False
        for edge in edges:
            s = edge[0]
            e = edge[1]
            s_anchor = find_anchor(s)
            e_anchor = find_anchor(e)
            if s_anchor is not None and e_anchor is not None and s_anchor == e_anchor:
                bad_tree = True
                break
            elif s_anchor is not None and e_anchor is not None and s_anchor != e_anchor:
                points_set[s_anchor] = merge_data_set(s_anchor, points_set[s_anchor].data, points_set[e_anchor].data)
                del points_set[e_anchor]
            elif s_anchor is not None and e_anchor is None:
                points_set[s_anchor] = merge_data_set(s_anchor, points_set[s_anchor].data, set([e]))
            elif s_anchor is None and e_anchor is not None:
                points_set[e_anchor] = merge_data_set(e_anchor, set([s]), points_set[e_anchor].data)
            elif s_anchor is None and e_anchor is None:
                points_set[s] = merge_data_set(s, set([s]), set([e]))
            else:
                raise ValueError('What happened !!! ')

        if bad_tree or len(points_set) > 1 or len((next((iter(points_set.values())))).data) < n:
            return False
        else:
            return True


if __name__ == "__main__":
    ts = GSolution()
    res = ts.validTree(5, [[0, 1], [0, 2], [2, 3], [2, 4]])
    print(res)
    res = ts.validTree(4, [[0, 1], [2, 3]])
    print(res)
    res = ts.validTree(4, [[2, 3], [1, 2], [1, 3]])
    print(res)
    # A = [100, 200, 300, 400]
    # for i, a in enumerate(A, 1):
    #     print(f" ==> {i} ==> {a} ==>")
    # ks = KSolution2()
    # res = ks.kEmptySlots([3, 9, 2, 8, 1, 6, 10, 5, 4, 7], 1)  # 6
    # print(res)
    # res = ks.kEmptySlots([1, 3, 2], 1)  # 2
    # print(res)
    # res = ks.kEmptySlots([1, 2, 3], 1)  # -1
    # print(res)
    # res = ks.kEmptySlots([6, 5, 8, 9, 7, 1, 10, 2, 3, 4], 2)  # 8
    # print(res)
    # res = ks.kEmptySlots([9, 1, 4, 2, 8, 7, 5, 3, 6, 10], 3)  # 5
    # print(res)
    # res = ks.kEmptySlots([6, 10, 7, 1, 9, 8, 4, 3, 5, 2], 3)  # 2
    # print(res)

    # big_data = ''
    # with open('D:\dev\workspace\python\googly\k-slot.txt', 'r') as k_slot_file:
    #     big_data = k_slot_file.read()
    # ll = [int(i) for i in big_data.split(',')]
    # res = ks.kEmptySlots_2(ll, 4973)  #
    # print(res)

    # print(licenseKeyFormatting("5F3Z-2e-9-w", 4))
    # print(next_high([21], 20))
    # print(next_high([19], 20))
    # print(next_high([19, 21], 20))
    # print(next_high([19, 21], 16))
    # print(next_high([16, 17, 21], 20))
    # res = get_permutations([1, 2, 3, 4, 1, 2, 3, 4], 2)
    # print(res)
    # ns = NewSolution()
    # # res = ns.nextClosestTime('19:34')
    # res = ns.nextClosestTime('23:59')
    # ns = NewSolution()
    # print(ns.next_time('19:34'))
    # print(ns.next_time('23:59'))
    # print(ns.next_time('13:55'))
    # print(ns.next_time('01:32'))
    # print(ns.next_time('13:55'))
    # print(res)
    # res = kEmptySlots([1, 2, 3], 1)
    # print(res)
    # parent = TreeNode(1)
    # left_child = TreeNode(1)
    # parent.left = left_child
    # res = isValidBST(parent)
    # print(res)
    # res = convert('PAYPALISHIRING',3)

    # res = repeatedStringMatch("a", "aa")  # 2
    # res = repeatedStringMatch("abababaaba","aabaaba")
    # res = repeatedStringMatch("bb", "bbbbbbb")
    # res = repeatedStringMatch("abaabaa", "abaababaab")
    # res = repeatedStringMatch("abcd", "abcdb")
    # res = repeatedStringMatch("abcd", "cdabcdacdabcda")
    # res = repeatedStringMatch("abcd", "bcdab")
    # print(res)

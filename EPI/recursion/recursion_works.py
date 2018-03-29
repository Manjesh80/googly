from collections import OrderedDict
from collections import namedtuple
from collections import Counter

PEGS_NAME = {0: "FROM"}


# 15.1 Compute Tower of Hanoi
def compute_tower_of_hanoi(num_rings):
    def print_pegs():
        print('*****************')
        print(f"FROM => {pegs[0]}")
        print(f"TO =>  {pegs[1]}")
        print(f"TEMP => {pegs[2]}")
        print('*****************')

    def compute_tower_hanoi_steps(num_rings_to_move, from_peg, to_peg, use_peg):
        if num_rings_to_move > 0:
            compute_tower_hanoi_steps(num_rings_to_move - 1, from_peg, use_peg, to_peg)
            pegs[to_peg].append(pegs[from_peg].pop())
            print_pegs()
            compute_tower_hanoi_steps(num_rings_to_move - 1, use_peg, to_peg, from_peg)
            print_pegs()

    NUM_PEGS = 3
    pegs = [list(reversed(range(1, num_rings + 1)))] + [[] for _ in range(1, NUM_PEGS)]
    compute_tower_hanoi_steps(num_rings, 0, 1, 2)
    return pegs


# 15.2 N Queen Placement
def n_queen_placement(n):
    def mark_checker(row, col, pos):
        for r_i in range(0, n):
            if r_i == row:
                for row_index in range(0, n):
                    checker[r_i][row_index] = pos
            for c_i in range(0, n):
                if c_i == col:
                    for col_index in range(0, n):
                        checker[col_index][c_i] = pos
        pass

    def place_queen(row, col):
        if row == n:
            valid_checker[0] += 1
        else:
            for pos in range(col, n):
                mark_checker(row, col)
                pass
        pass

    valid_checker = [0]
    checker = list([None] * n for _ in range(0, n))
    place_queen(0, 0)


def n_queen_placement_epi(n):
    def solve_n_queens(row_num):
        if row_num == n:
            result.append(list(col_placement))
            return
        for col_num in range(n):
            temp_results = []
            for i, placed_in_column in enumerate(col_placement[:row_num]):
                # if a queen is is already present in columns
                if abs(placed_in_column - col_num) not in (0, row_num - i):
                    temp_results.append(True)
                else:
                    temp_results.append(False)

            if all(temp_results):
                col_placement[row_num] = col_num
                solve_n_queens(row_num + 1)

    result, col_placement = [], [None] * n
    solve_n_queens(0)
    return result


def n_queen_placement_ext(n):
    def solve_n_queens(row_num):
        if row_num == n:
            result.append(list(col_placement))
            return
        for col_num in range(n):
            if row_num == 0:
                for i in range(0, len(col_placement)):
                    col_placement[i] = None
            disallowed_values = []
            prev_row_num = row_num - 1
            if prev_row_num >= 0:
                prev_row_placement = col_placement[prev_row_num]

                disallowed_values.extend([prev_row_placement - 1, prev_row_placement, prev_row_placement + 1])

            if col_num not in disallowed_values and col_num not in col_placement:
                col_placement[row_num] = col_num
                solve_n_queens(row_num + 1)

    result, col_placement = [], [None] * n
    solve_n_queens(0)
    return result


def n_queen_placement_back_track(n):
    def print_allowed_map():
        # print(" ********************* ")
        # for row in allowed_map:
        #     print(row)
        # print(" ********************* ")
        pass

    def block_entire_row_col(item, pos):
        for i in range(0, n):
            allowed_map[item][i].add(item)
            allowed_map[i][pos].add(item)

    def block_diagonal_by_item(item, pos):
        level = 0;
        for i in range(item + 1, n):
            level += 1
            if (pos + level) < n:
                allowed_map[i][pos + level].add(item)
            if (pos - level) >= 0:
                allowed_map[i][pos - level].add(item)

    def release_entire_row_col(item, pos):
        for i in range(0, n):
            if item in allowed_map[item][i]:
                allowed_map[item][i].remove(item)
            if item in allowed_map[i][pos]:
                allowed_map[i][pos].remove(item)

    def release_diagonal_by_item(item, pos):
        level = 0;
        for i in range(item + 1, n):
            level += 1
            if (pos + level) < n and item in allowed_map[i][pos + level]:
                allowed_map[i][pos + level].remove(item)
            if (pos - level) >= 0 and item in allowed_map[i][pos - level]:
                allowed_map[i][pos - level].remove(item)

    def position_available(item, pos):
        return True if len(allowed_map[item][pos]) == 0 else False

    def move_next(level):
        if level == (n):
            print(position_map)
            results.append(position_map)
            # print(allowed_map)
            return
        for pos in range(0, n):
            if position_available(level, pos):
                position_map[pos] = level
                block_entire_row_col(level, pos)
                block_diagonal_by_item(level, pos)
                # print(f" Before move {level} at {pos}")
                print_allowed_map()
                move_next(level + 1)
                release_entire_row_col(level, pos)
                release_diagonal_by_item(level, pos)
                # print(f" !!! After move {level} at {pos} !!! ")
                print_allowed_map()
                position_map[pos] = -1

    # allowed_map = [[set() for _ in range(0, n)] for _ in range(0, n)]
    allowed_map = []
    for i in range(0, n):
        allowed_map.append([])
        for j in range(0, n):
            allowed_map[i].append(set())

    position_map = [-1 for _ in range(0, n)]
    results = []
    move_next(0)
    return results


# 15.3 Generate permutations
def permutations_epi(A):
    def directed_permutations(i):
        if i == len(A):
            result.append(A.copy())
            return
        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            directed_permutations(i + 1)
            A[i], A[j] = A[j], A[i]

    result = []
    directed_permutations(0)
    return result


def permutations_for(A):
    result = []
    for i in range(0, len(A)):
        A[0], A[i] = A[i], A[0]
        for j in range(1, len(A)):
            A[1], A[j] = A[j], A[1]
            for k in range(2, len(A)):
                A[2], A[k] = A[k], A[2]
                result.append(A.copy())
                A[2], A[k] = A[k], A[2]
            A[1], A[j] = A[j], A[1]
        A[0], A[i] = A[i], A[0]
    return result


def permutations_epi_manjesh(A):
    def permute(i):
        if i == len(A):
            result.append(A.copy())
            return
        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            permute(i + 1)
            A[i], A[j] = A[j], A[i]

    result = []
    permute(0)
    return result


def next_permutation(A):
    inverse_point = len(A) - 2
    inverse_point_not_found = True
    while inverse_point > 0:
        if A[inverse_point] > A[inverse_point + 1]:
            inverse_point -= 1
        else:
            inverse_point_found = True

    if not inverse_point_found:
        return None
    # if inverse_point == 0:
    #     return None
    # Replace inverse point with next highest value in the sub-chain
    for i in reversed(range(inverse_point, len(A))):
        if A[inverse_point] < A[i]:
            A[inverse_point], A[i] = A[i], A[inverse_point]
            break
    # Sort the list after inverse point and append
    A[inverse_point + 1:] = sorted(A[inverse_point + 1:])
    return A


def permutations(A):
    result = []
    while True:
        A = next_permutation(A)
        print(A)
        if A:
            result.append(A.copy())
        else:
            break

    return result


# 15.4 Generate the power set
def generate_power_subset(input_set):
    def directed_power_set(to_be_selected, selected_so_far):
        if to_be_selected == len(input_set):
            power_set.append(list(selected_so_far))
            return
        directed_power_set(to_be_selected + 1, selected_so_far)
        directed_power_set(to_be_selected + 1, selected_so_far + [input_set[to_be_selected]])

    power_set = []
    directed_power_set(0, [])
    return power_set


def generate_power_subset_manjesh(A):
    def generate_power_subset_by_offset(anchor, offset, size):
        if offset + size > len(A):
            return
        else:
            results.append(list(A[anchor:anchor + 1] + A[offset: offset + size]))
        generate_power_subset_by_offset(anchor, offset + 1, size)

    def generate_power_subset_anchor(anchor):
        if anchor > len(A):
            return
        for size in range(1, len(A)):
            generate_power_subset_by_offset(anchor, anchor + 1, size)
        generate_power_subset_anchor(anchor + 1)

    results = []
    generate_power_subset_anchor(0)
    return results


def generate_power_subset_manjesh_new(A):
    def generate_power_subset(remaining, prefix):
        for i in range(0, len(remaining)):
            power_set.append(list(prefix + [remaining[i]]))
            generate_power_subset(remaining[i + 1:], prefix + [remaining[i]])

    power_set = []
    generate_power_subset(A, [])
    return power_set


def generate_power_subset_manjesh_recursion(input_set):
    def generate_power_subset(to_be_selected_index, up_to_know_powerset):
        if to_be_selected_index == len(input_set):
            return

        this_run_set = []
        for existing_set in up_to_know_powerset:
            new_set = existing_set.copy()
            new_set.add(input_set[to_be_selected_index])
            this_run_set.append(new_set)
        up_to_know_powerset.append(set(input_set[to_be_selected_index]))
        [up_to_know_powerset.append(s) for s in this_run_set]
        generate_power_subset(to_be_selected_index + 1, up_to_know_powerset)

    power_set = []
    generate_power_subset(0, power_set)
    power_set.append({})
    return power_set


# 15.5 Generate K subset
def generate_k_subset(A, k):
    def build_k_subset(anchor, offset):
        if anchor > len(A) or offset > len(A) - k + 1 or anchor + k > len(A):
            # print(f" END ==>  Anchor ==> {anchor} ==> offset {offset}")
            return
        else:
            # print('****')
            print(A[anchor:anchor + 1] + A[offset: offset + k - 1])
            # print(f" Added ==> Anchor ==> {anchor} ==> offset {offset}")
            # print('****')

        build_k_subset(anchor, offset + 1)
        build_k_subset(anchor + 1, anchor + 1)

    results = []
    build_k_subset(0, 0)
    return results


def generate_k_subset_new(A, k):
    def build_k_subset_move_offset(anchor, offset):
        if offset + k - 1 > len(A):
            return
        else:
            results.append(list(A[anchor:anchor + 1] + A[offset: offset + k - 1]))
        build_k_subset_move_offset(anchor, offset + 1)

    def build_k_subset_move_anchor(anchor):
        if anchor > len(A) - k:
            return
        build_k_subset_move_offset(anchor, anchor + 1)
        build_k_subset_move_anchor(anchor + 1)

    results = []
    build_k_subset_move_anchor(0)
    return results


# 15.6 Matching parantheses count
def num_parantheses_count(N):
    parentheses_dict = {1: 0, 2: 1}
    for i in range(3, N + 1):
        i_count = 0
        for j in range(1, i // 2 + 1):
            i_count += parentheses_dict[j] + parentheses_dict[i - j]
            # i_count = max(i_count, parentheses_dict[j] + parentheses_dict[i - j])
        parentheses_dict[i] = i_count
    return parentheses_dict[N]


def generate_balanced_parantheses(num_pairs):
    def directed_generate_balanced_parantheses(num_of_left_paren, num_of_right_paren, valid_prefix, result=[]):
        print(
            f" Calling method ==> Left ==> {num_of_left_paren} Right ==> {num_of_right_paren} Prefix is ==> {valid_prefix}")
        if num_of_left_paren > 0:
            directed_generate_balanced_parantheses(num_of_left_paren - 1, num_of_right_paren, valid_prefix + '[')
        if num_of_left_paren < num_of_right_paren:
            directed_generate_balanced_parantheses(num_of_left_paren, num_of_right_paren - 1, valid_prefix + ']')
        if not num_of_right_paren:
            result.append(valid_prefix)
        return result

    return directed_generate_balanced_parantheses(num_pairs, num_pairs, '')


def generate_balanced_parantheses_epi(num_pairs):
    def directed_generate_balanced_parantheses(num_of_left_paren, num_of_right_paren, valid_prefix, result=[]):
        if num_of_left_paren > 0:
            directed_generate_balanced_parantheses(num_of_left_paren - 1, num_of_right_paren, valid_prefix + '[')
        if num_of_left_paren < num_of_right_paren:
            directed_generate_balanced_parantheses(num_of_left_paren, num_of_right_paren - 1, valid_prefix + ']')
        if not num_of_right_paren:
            result.append(valid_prefix)
        return result

    return directed_generate_balanced_parantheses(num_pairs, num_pairs, '')


def permutation_dup_not_supported(input):
    dict = OrderedDict()
    for k in input:
        if k not in dict:
            dict[k] = 0
        dict[k] += 1

    def create_permutation(level):
        if level == len(input):
            results.append(current_permutation.copy())
            return

        for i in range(0, len(input)):
            to_add = input[i];
            if dict[to_add] > 0:
                dict[to_add] -= 1
                current_permutation.append(to_add)
                create_permutation(level + 1)
                current_permutation.remove(to_add)
                dict[to_add] += 1

    results = []
    current_permutation = []
    create_permutation(0)
    return results


def permutation(input):
    dict = OrderedDict()
    for k in input:
        if k not in dict:
            dict[k] = 0
        dict[k] += 1

    def dict_value_at_pos(pos):
        temp_list = list(dict.items())
        return temp_list[pos]

    def create_permutation(level):
        if level == len(input):
            results.append(current_permutation.copy())
            return

        for i in range(0, len(dict)):
            to_add, count = dict_value_at_pos(i)
            if count > 0:
                dict[to_add] -= 1
                current_permutation.append(to_add)
                create_permutation(level + 1)
                current_permutation.remove(to_add)
                dict[to_add] += 1

    results = []
    current_permutation = []
    create_permutation(0)
    return results


def generate_permutation(values):
    root_counter = Counter(values)

    def gen_cyclic_permutation(curr_counter):
        if len(current_perm) == len(values):
            result.append(current_perm.copy())
            return
        for key in curr_counter.keys():
            if curr_counter[key] == 0:
                continue
            current_perm.append(key)
            curr_counter[key] -= 1
            gen_cyclic_permutation(curr_counter)
            curr_counter[key] += 1
            current_perm.pop()

    result = []
    current_perm = []
    gen_cyclic_permutation(root_counter)
    return result


# 15.7 Generate palindromic decompostions
def palindrome_partitioning(input):
    def directed_palindrome_partitioning(offset, partial_partition):
        if offset == len(input):
            result.append(list(partial_partition))
            return

        for i in range(offset + 1, len(input) + 1):
            prefix = input[offset:i]
            if prefix == prefix[::-1]:
                directed_palindrome_partitioning(i, partial_partition + [prefix])

    result = []
    directed_palindrome_partitioning(0, [])
    return result


class BTNode:
    def __init__(self, name=None, left=None, right=None):
        self.name = name
        self.left = left
        self.right = right


# 15.8 Generate binary trees
def create_bt_tree_node(num_nodes):
    if num_nodes == 0:
        return [None]
    result = []
    for num_left_nodes in range(num_nodes):
        num_right_nodes = num_nodes - 1 - num_left_nodes
        left_subtrees = create_bt_tree_node(num_left_nodes)
        right_subtrees = create_bt_tree_node(num_right_nodes)
        result += [BTNode(0, left, right)
                   for left in left_subtrees
                   for right in right_subtrees]

    return result


def get_bt_possibility_count(n):
    node_len_count = {0: 1, 1: 1}
    for i in range(2, n + 1):
        cur_node_len_count = 0
        for j in range(0, i):
            cur_node_len_count += node_len_count[j] * node_len_count[i - j - 1]
        node_len_count[i] = cur_node_len_count
    return node_len_count[n]


# 15.10 Compute Gray Code
def grey_code(bits):
    n = 2 ** bits

    def directed_gray_code(level):
        if level == n:
            if abs(bin(current_sequence[0]).count("1") - bin(current_sequence[-1]).count("1")) == 1:
                results.append(current_sequence.copy())
            return

        for i in range(0, n):
            if i not in current_sequence:
                if level > 0:
                    if abs(bin(i).count("1") - bin(current_sequence[level - 1]).count("1")) == 1:
                        current_sequence[level] = i
                    else:
                        continue
                else:
                    current_sequence[level] = i
                directed_gray_code(level + 1)
                current_sequence[level] = -1

    current_sequence = [-1 for _ in range(n)]
    results = []
    directed_gray_code(0)
    return results


# 15.11 Generate Palindromic sequences
# res = gen_palindromic_sequences_tushar(list('14141'))
def gen_palindromic_sequences_tushar(values):
    def directed_palindromic_sequences(level):
        if level == len(values):
            result.append(current_result.copy())
            return
        for i in range(level, len(values) + 1):
            current_str = ''.join(values[level: i + 1])
            if current_str == current_str[::-1]:  # Palindrome so continue
                current_result.append(current_str)
                directed_palindromic_sequences(i + 1)
                current_result.pop()

    result = []
    current_result = []
    directed_palindromic_sequences(0)
    return result


# 15.12 Generate matched parantheses
# res = generate_match_parantheses(4)
def generate_match_parantheses(total_pair):
    def generated_directed_parantheses(left_remaining, right_remaining, prefix):
        if not left_remaining and not right_remaining:
            result.append("".join(prefix.copy()))
            return
        if left_remaining > 0:
            prefix.append('[')
            generated_directed_parantheses(left_remaining - 1, right_remaining, prefix)
            prefix.pop()

        if right_remaining > left_remaining:
            prefix.append(']')
            generated_directed_parantheses(left_remaining, right_remaining - 1, prefix)
            prefix.pop()

    result = []
    generated_directed_parantheses(total_pair, total_pair, [])
    return result


# 15.13 Generate K - Subset
def generate_k_subset(values, combinations):
    def directed_k_subset(anchor, current_level, prefix):
        if current_level == combinations:
            result.append(prefix.copy())
            return

        prefix.append(values[anchor])
        for i in range(anchor + 1, len(values)):
            prefix.append(values[i])
            directed_k_subset(i, current_level + 1, prefix)
            prefix.pop()
        prefix.pop()

    result = []
    directed_k_subset(0, 0, [])
    return result


# 15.13 Generate K - Subset
def generate_k_subset_new(values, combinations):
    def directed_k_subset(anchor, current_level, prefix):
        if current_level == combinations:
            result.append(prefix.copy())
            return

        if anchor < combinations + 1:
            prefix.append(values[anchor])
            for i in range(anchor + 1, len(values)):
                directed_k_subset(i, current_level + 1, prefix)
            prefix.pop()

    result = []
    directed_k_subset(0, 0, [])
    return result


def generate_k_subset_rec(values, combinations):
    def directed_k_subset(anchor, level, prefix):
        if len(prefix) == combinations:
            result.append(prefix.copy())
            return
        for i in range(level, combinations + 1):
            prefix.append(values[i])
            for j in range(i + 1, len(values)):
                prefix.append(values[j])
                directed_k_subset(j, i + 1, prefix)
                prefix.pop()
            prefix.pop()

    result = []
    directed_k_subset(0, 0, [])
    return result


def generate_k_subset_rec_new(values, combinations):
    def directed_k_subset(level, prefix):
        if len(prefix) == combinations:
            result.append(prefix.copy())
            return

        for i in range(level, combinations + 1):
            prefix.append(values[i])
            for j in range(i + 1, len(values)):
                prefix.append(values[j])
                directed_k_subset(i + 1, prefix)
                prefix.pop()
            prefix.pop()

    result = []
    directed_k_subset(0, [])
    return result


def generate_k_subset_rec_new(values, combinations):
    def directed_subset(level, anchor, prefix):
        if len(prefix) == combinations:
            result.append(prefix.copy())
            return

        for i in range(anchor + 1, len(values)):
            prefix.append(values[i])
            directed_subset(level + 1, i, prefix)
            prefix.pop()

    result = []
    directed_subset(0, -1, [])
    return result


def generate_k_subset_rec_new_2(values, combinations):
    def directed_subset(anchor, prefix):
        if len(prefix) == combinations:
            result.append(prefix.copy())
            return

        for i in range(anchor + 1, len(values)):
            prefix.append(values[i])
            directed_subset(i, prefix)
            prefix.pop()

    result = []
    directed_subset(-1, [])
    return result


def combinations(n, k):
    def directed_combinations(offset, partial_combination):
        if len(partial_combination) == k:
            result.append(list(partial_combination))
            return

        num_remaining = k - len(partial_combination)
        i = offset
        while i <= n and num_remaining <= n - i + 1:
            directed_combinations(i + 1, partial_combination + [i])
            i += 1

    result = []
    directed_combinations(1, [])
    return result


def kadens_max_sub_array(A):
    if not A:
        return -1
    current_max = global_max = A[0]
    for i in range(1, len(A) - 1):
        current_max = max(A[i], current_max + A[i])
        global_max = max(global_max, current_max)
    return global_max


class Edge(object):
    def __init__(self, name, weight, start, end):
        self.name = name
        self.weight = weight
        self.start = start
        self.end = end


class SimpleNode:
    def __init__(self, name):
        self.name = name
        self.edges = []

    def add_edges(self, edges):
        self.edges = edges


def build_simple_diameter_tree():
    nodeA = SimpleNode('A')
    nodeB = SimpleNode('B')
    nodeC = SimpleNode('C')
    nodeD = SimpleNode('D')

    nodeE = SimpleNode('E')
    nodeF = SimpleNode('F')
    nodeG = SimpleNode('G')

    edge_A_B = Edge('AB', 10, nodeA, nodeB)
    edge_A_C = Edge('AC', 1, nodeA, nodeC)
    edge_A_D = Edge('AD', 1, nodeA, nodeD)

    edge_C_E = Edge('CE', 3, nodeC, nodeE)
    edge_C_F = Edge('CF', 1, nodeC, nodeF)
    edge_C_G = Edge('CG', 3, nodeC, nodeG)

    nodeA.add_edges([edge_A_B, edge_A_C, edge_A_D])
    nodeC.add_edges([edge_C_E, edge_C_F, edge_C_G])
    return nodeA


# 15.11 Calculate tree diameter
def tree_diameter(root):
    HD = namedtuple('HD', ('name', 'height', 'diameter'))

    def calculate_tree_diameter(node):
        if not node.edges or len(node.edges) == 0:
            return HD(name=node.name, height=0, diameter=0)
        edges_h_d = []
        child_originals = []
        for edge in node.edges:
            h_d = calculate_tree_diameter(edge.end)
            child_originals.append(h_d)
            edges_h_d.append(HD(name=edge.name, height=h_d.height + edge.weight, diameter=h_d.diameter + edge.weight))

        max_height = max(edges_h_d, key=lambda mh: mh.height)[1]
        edges_h_d.sort(key=lambda mh: mh.height, reverse=True)
        total_diameter_this_node = sum([hd.height for hd in edges_h_d[:2]])
        total_diameter_child = max(child_originals, key=lambda mh: mh.diameter)[2]
        global_max_diameter[0] = max(global_max_diameter[0], total_diameter_this_node, total_diameter_child)
        res = HD(name=node.name, height=max_height, diameter=total_diameter_this_node)
        print(res)
        return res

    global_max_diameter = [0]
    return global_max_diameter, calculate_tree_diameter(root)


if __name__ == "__main__":
    # res = generate_power_subset(['A', 'B', 'C'])
    # root = build_simple_diameter_tree()
    # res = tree_diameter(root)
    res = generate_permutation(['A', 'A', 'B', 'C'])
    # res = permutation(['A', 'A', 'B', 'B'])
    print(res)
    print(len(res))

# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments
# Comments

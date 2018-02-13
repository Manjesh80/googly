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


if __name__ == "__main__":
    res = num_parantheses_count(4)
    print(res)

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

from collections import defaultdict
from collections import namedtuple
from enum import Enum


# 16.1 different score combinations
def num_comb_final_score(final_score, individual_play_score):
    num_comb_for_score = [[1] + [0] * final_score for _ in individual_play_score]

    for i in range(len(individual_play_score)):
        for j in range(1, final_score + 1):
            without_this_play = 0
            with_this_play = 0
            if i > 0:
                without_this_play = num_comb_for_score[i - 1][j]
            if j >= individual_play_score[i]:
                with_this_play = num_comb_for_score[i][j - individual_play_score[i]]
            num_comb_for_score[i][j] = without_this_play + with_this_play

    print('***********************')
    for score in num_comb_for_score:
        print(score)
    print('***********************')
    return num_comb_for_score[-1][-1]


class EdOp(Enum):
    Unprocessed = 0
    Delete = 1
    Insert = 2
    Substitute = 3
    Copy = 4


# 16.2 Minimum edit distance
def minimum_edit_distance(F, T):
    def find_edit(i, j):
        if result[i][j][1] != EdOp.Unprocessed:
            return
        if i == 0:
            print(f"Processing  Zero Logic I ==> {i} <==>  j ==> {j} ")
            result[i][j] = (j, EdOp.Insert)
            return
        if j == 0 and i != 0:
            print(f" *************** ")
            print(f"Processing  Zero Logic I ==> {i} <==>  j ==> {j} ")
            result[i][j] = (i, EdOp.Delete)
            return
        find_edit(i - 1, j)
        find_edit(i, j - 1)
        if result[i][j][1] == EdOp.Unprocessed:
            result[i][j] = (1, EdOp.Insert)
            print(f"Processing  Main Logic I ==> {i} <==>  j ==> {j} ")
            top_value = result[i - 1][j][0]
            left_value = result[i][j - 1][0]
            diagonal_value = result[i - 1][j - 1][0]
            are_values_same = (F[i] == T[j])

            if top_value < left_value and top_value < diagonal_value:
                result[i][j] = (top_value + 1, EdOp.Delete)
            elif left_value < top_value and left_value < diagonal_value:
                result[i][j] = (left_value + 1, EdOp.Insert)
            else:
                if are_values_same:
                    result[i][j] = (diagonal_value, EdOp.Copy)
                else:
                    result[i][j] = (diagonal_value + 1, EdOp.Substitute)

    result = [[(0, EdOp.Unprocessed)] * len(T) for _ in F]
    find_edit(len(F) - 1, len(T) - 1)
    return result


# 16.3 Different ways to traverse Matrix
class Node_2D:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.route_down = None
        self.route_right = None

    def add_route_down(self, node):
        self.route_down = node

    def add_route_right(self, node):
        self.route_right = node

    def get_id(self):
        return self.i, self.j

    def is_end(self):
        return self.route_down is None and self.route_right is None

    def get_down(self):
        return self.i + 1, self.j

    def get_right(self):
        return self.i, self.j + 1

    def get_down_node(self):
        return self.route_down

    def get_right_node(self):
        return self.route_right


def traverse_2d_array(A, start_i, start_j):
    def traverse_node(node, prefix):
        if not node:
            return
        if node.is_end():
            prefix.append(node.get_id())
            paths.append(prefix)
            return
        traverse_node(node.get_down_node(), list(prefix + [node.get_id()]))
        traverse_node(node.get_right_node(), list(prefix + [node.get_id()]))

    def process_node(node):
        node.add_route_down(routes_dict.get(node.get_down()))
        node.add_route_right(routes_dict.get(node.get_right()))
        routes_dict[node.get_id()] = node

    routes_dict = defaultdict()
    level = 0
    for anchor in reversed(range(0, len(A))):
        for i in reversed(range(anchor + 1, len(A))):
            process_node(Node_2D(anchor, i))
            process_node(Node_2D(i, anchor))
        process_node(Node_2D(anchor, anchor))
        level += 1
    start_node = routes_dict[(start_i, start_j)]
    paths = []
    traverse_node(start_node, [])
    return paths


def num_of_ways(r, c):
    def compute_num_of_ways_to_xy(x, y):
        if x == y == 0:
            return 1
        if number_of_ways[x][y] == 0:
            ways_top = 0 if x == 0 else compute_num_of_ways_to_xy(x - 1, y)
            ways_left = 0 if y == 0 else compute_num_of_ways_to_xy(x, y - 1)
            number_of_ways[x][y] = ways_top + ways_left
        return number_of_ways[x][y]

    number_of_ways = [[0] * c for _ in range(r)]


# 16.6 Knapsack problem

Item = namedtuple('Item', ('weight', 'value'))


def max_knap_sack(WV, M):
    def knap_sack(item_count, W):
        for cw in range(w):
            pass

    OIW = [[-1] * (len(WV) + 1)] * (M + 1)

    for item in len(WV):
        for w in range(M):
            knap_sack(item, w)

    return OIW


def optimum_subject_to_capacity(items, capacity):
    def optimum_subject_to_item_and_capacity(item_count, available_capacity):
        if item_count < 0:
            return 0
        if V[item_count][available_capacity] != -1:
            with_out_current_time = optimum_subject_to_item_and_capacity(
                item_count - 1, available_capacity)
            with_current_time = (0 if available_capacity < items[item_count].weight else
            (items[item_count].value + optimum_subject_to_item_and_capacity(
                item_count - 1, available_capacity - items[item_count].weight
            )))
            V[item_count][available_capacity] = max(with_current_time, with_out_current_time)
        return V[item_count][available_capacity]

    V = [[-1] * (capacity + 1) for _ in items]
    return optimum_subject_to_item_and_capacity(len(items) - 1, capacity)


def optimum_subject_to_capacity_new(items, capacity):
    def optimum_subject_to_item_and_capacity(item_count, available_capacity):
        if item_count < 0:
            return 0

        if V[item_count][available_capacity] != -1:
            with_out_current_time = optimum_subject_to_item_and_capacity(
                item_count - 1, available_capacity)

            with_current_time = 0
            if available_capacity > items[item_count].weight:
                with_current_time = \
                    items[item_count].value + optimum_subject_to_item_and_capacity(
                        item_count - 1, available_capacity - items[item_count].weight)

        V[item_count][available_capacity] = max(with_current_time, with_out_current_time)

    V = [[-1] * (capacity + 1) for _ in items]
    return optimum_subject_to_item_and_capacity(len(items) - 1, capacity)


def knapsack_value(items, capacity):
    def grab_and_go(cur_item_len, cur_weight):
        if cur_item_len < 0 or cur_weight < 0:
            return
        grab_and_go(cur_item_len - 1, cur_weight)
        grab_and_go(cur_item_len, cur_weight - 1)
        if result_cache[cur_item_len][cur_weight] == -1:
            current_item_weight, current_item_value = items[cur_item_len]
            max_value = current_item_value
            last_value_item = result_cache[cur_item_len - 1][cur_weight]
            if current_item_weight > cur_weight:
                result_cache[cur_item_len][cur_weight] = max(0, last_value_item)
            else:
                k_list = list(range(1, cur_weight))
                for k in k_list:
                    left_value = result_cache[cur_item_len][k]
                    right_value = result_cache[cur_item_len][cur_weight - k]
                    cache_hit[0] += 1
                    combo_value = left_value + right_value
                    max_value = max(max_value, last_value_item, combo_value)
                result_cache[cur_item_len][cur_weight] = max_value

    result_cache = [[-1] * (capacity + 1) for _ in items]
    cache_hit = [0]
    grab_and_go(len(items) - 1, capacity)
    print(f"Cache hit => {cache_hit}")
    return result_cache


def knapsack_value_updated(items, capacity):
    def grab_and_go(cur_item_len, cur_weight):
        if cur_item_len < 0 or cur_weight < 0:
            return
        grab_and_go(cur_item_len - 1, cur_weight)
        grab_and_go(cur_item_len, cur_weight - 1)
        if result_cache[cur_item_len][cur_weight] == -1:
            current_item_weight, current_item_value = items[cur_item_len]
            if current_item_weight > capacity or cur_weight < current_item_weight:
                result_cache[cur_item_len][cur_weight] = max(0, result_cache[cur_item_len - 1][cur_weight])
            else:
                prev_row_same_weight = max(0, result_cache[cur_item_len - 1][cur_weight])
                prev_row_reduced_weight = max(0, result_cache[cur_item_len - 1][cur_weight - current_item_weight])
                result_cache[cur_item_len][cur_weight] = max(prev_row_same_weight, (
                        prev_row_reduced_weight + current_item_value
                ))

    result_cache = [[-1] * (capacity + 1) for _ in items]
    cache_hit = [0]
    grab_and_go(len(items) - 1, capacity)
    print(f"Cache hit => {cache_hit}")
    return result_cache


# 16.X Rod cutting problem
def optimize_rod_cut(table, l):
    def get_optimal_price(current_cut):
        if current_cut > 0:
            get_optimal_price(current_cut - 1)
        if cache.get(current_cut) is not None:
            return cache.get(current_cut)
        max_result = 0 if table.get(current_cut) is None else table.get(current_cut)
        for k in range(1, current_cut - 1):
            max_result = max(max_result, (cache.get(current_cut - k) + cache.get(k)))
        cache[current_cut] = max_result
        return max_result

    cache = defaultdict(None)
    cache[0] = 0
    result = get_optimal_price(l)
    return result


# 16.9 pickup coins for maximum gain
def pick_up_coins_fox_max(coins):
    def max_profit_for_range(a, b):
        if a > b:
            return 0

        if max_matrix[a][b] == 0:
            i_select_a_other_select_a = max_profit_for_range(a + 2, b)
            i_select_a_other_select_b = max_profit_for_range(a + 1, b - 1)
            revenue_for_a = coins[a] + min(i_select_a_other_select_a, i_select_a_other_select_b)

            i_select_b_other_select_a = max_profit_for_range(a + 1, b - 1)
            i_select_b_other_select_b = max_profit_for_range(a, b - 2)
            revenue_for_b = coins[b] + min(i_select_b_other_select_a, i_select_b_other_select_b)

            max_matrix[a][b] = max(revenue_for_a, revenue_for_b)
        return max_matrix[a][b]

    max_matrix = [[0] * len(coins) for _ in coins]
    max_profit_for_range(0, len(coins) - 1)
    return max_matrix


# 16.10 Number of moves to climb the stairs
def num_of_moves_to_climb_stairs(H, S):
    def num_of_moves_to_top(T):
        if T <= 1:
            return 1
        if num_of_ways_to_height[T] == 0:
            num_of_ways = 0
            for i in range(1, S + 1):
                num_of_ways += num_of_moves_to_top(T - i)
            num_of_ways_to_height[T] = num_of_ways
        return num_of_ways_to_height[T]

    num_of_ways_to_height = [0] * (H + 1)
    num_of_moves_to_top(H)
    return num_of_ways_to_height[-1]


# 16.12 Longest non decreasing subset
def longest_non_decreasing_subset(A):
    max_length = [1] * len(A)
    for i in range(1, len(A)):
        max_j_which_is_lesser_than_i = []
        for j in range(i):
            if A[i] > A[j]:
                max_j_which_is_lesser_than_i.append(max_length[j])
        max_length[i] = max(max(max_j_which_is_lesser_than_i) + 1, max_length[i])
    return max(max_length)


# 16.11 Minimum messiness TODO
def minimum_mesiness(words, line_length):
    num_remaining_blanks = line_length - len(words[0])
    min_mesiness = ([num_remaining_blanks ** 2] + [float('inf')] * (len(words) - 1))
    for i in range(1, len(words)):
        num_remaining_blanks = line_length - len(words[i])
        min_mesiness[i] = min_mesiness[i - 1] + num_remaining_blanks ** 2
        for j in reversed(range(i)):
            num_remaining_blanks -= len(words[j]) + 1
            if num_remaining_blanks < 0:
                break
            first_j_mesiness = 0 if j - 1 < 0 else min
            pass


# 16.4 derive combination
def compute_binomial_coefficents(N, K):
    def derive_combination(x, y):
        if y in (0, x):
            return 1
        if results[x][y] == 0:
            with_y = derive_combination(x - 1, y)
            without_y = derive_combination(x - 1, y - 1)
            results[x][y] = with_y + without_y
        return results[x][y]

    results = [[0] * (K + 1) for _ in range(N + 1)]
    derive_combination(N, K)
    return results[-1][-1]


def basic_recursion(x, y):
    def set_values(i, j):
        if i == 0 or j == 0:
            result[i][j] = 1
            return
        set_values(i - 1, j)
        set_values(i, j - 1)
        if result[i][j] == -1:
            result[i][j] = 9

    result = [[-1] * x for _ in range(y)]
    set_values(x - 1, y - 1)
    return result


# 16.7 Dictionary decomposition
def dictionary_decomposition(lookup, word):
    def decompose_word(i, j):
        if result[i][j] != -1 or i < 0 or j < 0:
            return
        # if i == 0:
        #     result[i][j] = 0
        #     return
        # if j == 0 and i != 0:
        #     result[i][j] = 1
        #     return
        decompose_word(i - 1, j)
        decompose_word(i, j - 1)
        if result[i][j] == -1:
            final_result = 0
            for k in range(0, j + 1):
                sub_subword = word[k:j + 1]
                if sub_subword in lookup[:i + 1]:
                    if result[i - 1][j - len(sub_subword)] != 0 or result[i][j - len(sub_subword)] != 0:
                        final_result = 1
                        break
            result[i][j] = final_result

    result = [([-1] * len(word)) for _ in range(len(lookup))]
    # result[0][0] = 1
    decompose_word(len(lookup) - 1, len(word) - 1)
    return result


if __name__ == "__main__":
    # res = dictionary_decomposition(['car', 'bar'], 'carbar')
    res = dictionary_decomposition(['a', 'man', 'plan', 'canal'], 'amanaplanacanal')
    # res = basic_recursion(3, 3)
    for r in res:
        print(r)

# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment

from collections import defaultdict
from collections import namedtuple


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


if __name__ == "__main__":
    # items = [(5, 60), (3, 50), (4, 70), (2, 30)]
    # res = knapsack_value(items, 5)
    items = [(5, 5), (6, 4), (8, 7), (4, 7)]
    res = knapsack_value_updated(items, 13)
    print(res)

    # VW = [Item(5, 5), Item(6, 4), Item(8, 7), Item(4, 7)]
    # max_knap_sack(VW, 13)
    # # A = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # A = [[*range(i, i + 3)] for i in range(1, 9, 3)]
    # # A = [[*range(i, i + 10)] for i in range(1, 100, 10)]
    # paths = traverse_2d_array(A, 0, 0)
    # print(len(paths))

    # (2,2)
    # (2,1)
    # (1,2)
    # (1,1)
    # (2,0)
    # (1,0)
    # (0,2)
    # (0,1)
    # (0,0)

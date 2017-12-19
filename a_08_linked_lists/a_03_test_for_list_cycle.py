class Node:
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node


def generate_list(start, end, func, cycle_at=-1):
    end += 1
    root_node = None
    cur_node = None
    cycle_node = None
    for i in range(start, end):
        if root_node is None:
            root_node = Node(data=func(i), next_node=cur_node)
            cur_node = root_node
            continue
        else:
            new_node = Node(data=func(i), next_node=None)
            cur_node.next_node = new_node
            cur_node = new_node

        if cycle_at > 0 and cycle_at == i:
            cycle_node = cur_node

    if cycle_node is not None:
        cur_node.next_node = cycle_node

    return root_node


def print_list(root_node):
    while root_node is not None:
        print(f"Current element is ==> {root_node.data}")
        root_node = root_node.next_node


def has_cycle(ml):
    slow_ml = ml
    print(f"Slow Node Start ==> {slow_ml.data}")
    fast_ml = ml.next_node
    print(f"Fast Node Start ==> {fast_ml.data}")

    cycle_len = 0
    while slow_ml != fast_ml:
        slow_ml = slow_ml.next_node
        fast_ml = fast_ml.next_node.next_node
        cycle_len += 1
        # print(" ************** ")
        # print(f"Cycle ==> {cycle_len}")
        # print(f"Slow Node Start ==> {slow_ml.data}")
        # print(f"Fast Node Start ==> {fast_ml.data}")
        # print(" ************** ")

    cycle_len = cycle_len + 1
    print(f"Cycle length ==> {cycle_len}")

    finder_ml = ml
    while cycle_len != 0:
        cycle_len = cycle_len - 1
        finder_ml = finder_ml.next_node

    while ml != finder_ml:
        ml = ml.next_node
        finder_ml = finder_ml.next_node
        # print(" ************** ")
        # print(f"Master Node ==> {ml.data}")
        # print(f"Finder Node ==> {finder_ml.data}")
        # print(" ************** ")
    print(f"Cycle starts at ==> {finder_ml.data}")


if __name__ == '__main__':
    lst = generate_list(1, 11, lambda x: x, cycle_at=3)
    has_cycle(lst)

class Node:
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

    def __len__(self):
        list_len = 1
        next_node = self.next_node
        while next_node is not None:
            next_node = next_node.next_node
            list_len += 1
        return list_len

    def tail(self):
        tail_node = self
        while tail_node.next_node is not None:
            tail_node = tail_node.next_node
        return tail_node


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

    return root_node, cur_node


def print_list(root_node):
    while root_node is not None:
        print(f"Current element is ==> {root_node.data}")
        root_node = root_node.next_node


def do_list_overlap(l1, l2):
    l1_len = len(l1)
    l2_len = len(l2)

    ordered_lst = [(l1, l1_len), (l2, l2_len)] if l1_len > l2_len else [(l2, l2_len), (l1, l1_len)]
    big_lst = ordered_lst[0][0]
    big_lst_len = ordered_lst[0][1]
    small_lst = ordered_lst[1][0]
    small_lst_len = ordered_lst[1][1]

    if big_lst.tail() is small_lst.tail():
        print(f" Lists overlap ")
    else:
        print(f" Lists Don't overlap ")

    list_len_diff = big_lst_len - small_lst_len
    while list_len_diff != 0:
        list_len_diff -= 1
        big_lst = big_lst.next_node

    while big_lst != small_lst:
        big_lst = big_lst.next_node
        small_lst = small_lst.next_node

    print(f"Joining node is {big_lst.data}")


if __name__ == '__main__':
    tail_lst_head, tail_lst_tail = generate_list(9, 12, lambda x: x)
    big_lst_head, big_lst_tail = generate_list(1, 6, lambda x: x)
    big_lst_tail.next_node = tail_lst_head
    small_lst_head, small_lst_tail = generate_list(7, 8, lambda x: x)
    small_lst_tail.next_node = tail_lst_head
    print_list(big_lst_head)
    print("*****************")
    print_list(small_lst_head)
    print("*****************")

    do_list_overlap(big_lst_head, small_lst_head)

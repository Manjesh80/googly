class Node:
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node


def generate_list(start, end, func):
    root_node = None
    cur_node = None
    for i in range(start, end):
        if root_node is None:
            root_node = Node(data=func(i), next_node=cur_node)
            cur_node = root_node
            continue
        else:
            new_node = Node(data=func(i), next_node=None)
            cur_node.next_node = new_node
            cur_node = new_node
    return root_node


def print_list(root_node):
    while root_node is not None:
        print(f"Current element is ==> {root_node.data}")
        root_node = root_node.next_node


def reverse_sub_final(ml, start_index, end_index):
    if start_index == end_index:
        return ml

    dummy_head = Node(0, ml)
    mover = 1
    while mover < start_index - 1:
        mover += 1
        ml = ml.next_node

    sl = ml.next_node
    mover += 1
    while mover <= end_index:
        mover += 1
        temp = sl.next_node  # 4 , 5
        sl.next_node = temp.next_node  # 5 , 6
        temp.next_node = ml.next_node  # 3 , 4
        ml.next_node = temp

    print_list(dummy_head.next_node)


def reverse_sub_new(ml, start_index, end_index):
    if start_index == end_index:
        return ml

    dummy_head = Node(0, ml)
    mover = 1
    while mover < start_index:
        mover += 1
        ml = ml.next_node

    # Reverse sublist Iter
    sl = ml.next_node
    mover += 1
    while mover <= end_index:
        mover += 1
        print(" ********************* ")
        temp = sl.next_node  # 4 , 5
        print(f" Current temp ==> {temp.data}")
        sl.next_node = temp.next_node  # 5 , 6
        print(f" sl.next_node ==> {sl.next_node.data}")
        temp.next_node = ml.next_node  # 3 , 4
        print(f" temp.next_node ==> {temp.next_node.data}")
        ml.next_node = temp
        print(f" ml.next_node==> {ml.next_node.data}")
        print(" ********************* ")

    print_list(dummy_head.next_node)


def reverse_sub_list(master_list, start_index, end_index):
    if start_index == end_index:
        return master_list

    dummy_head = Node(0, master_list)
    sub_list_head = dummy_head
    mover = 1
    while mover < start_index:
        mover += 1
        sub_list_head = sub_list_head.next_node

    # Reverse sublist Iter
    sub_list_iter = sub_list_head.next_node
    while mover <= end_index:
        print(" ********************* ")
        print(f" Current mover ==> {mover}")
        mover += 1
        temp = sub_list_iter.next_node  # 4
        print(f" Current temp ==> {temp.data}")
        sub_list_iter.next_node = temp.next_node  # 5
        print(f" Current mover ==> {sub_list_iter.next_node.data}")
        temp.next_node = sub_list_head.next_node
        print(f" Current mover ==> {sub_list_iter.next_node.data}")
        sub_list_head.next_node = temp
        print(" ********************* ")

    print_list(dummy_head.next_node)


if __name__ == "__main__":
    lst = generate_list(1, 11, lambda x: x)
    print_list(lst)
    print('*********')
    reverse_sub_final(lst, 3, 6)

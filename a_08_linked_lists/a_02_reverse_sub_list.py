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


def reverse_sub_list(root_node, start_index, end_index):
    # get the trigger point
    head_list_end = root_node
    mover = 1
    while mover < start_index - 1:
        mover += 1
        head_list_end = head_list_end.next_node

    # create sublist from flip to end
    sub_list_iter = head_list_end.next_node

    while mover <= end_index:
        temp = sub_list_iter.next_node # sub_list_iter=4,  temp = 5 , temp.next = 6
        sub_list_iter.next = temp
        temp.next =
        # 5 to 4


if __name__ == "__main__":
    lst = generate_list(1, 11, lambda x: x)
    print('*********')
    reverse_sub_list(lst, 4, 8)

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


def generate_list(start, end, func, cycle_at=-1, get_at=-1):
    end += 1
    root_node = None
    cur_node = None
    cycle_node = None
    get_node = None
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

        if get_at > 0 and get_at == i:
            get_node = cur_node

    if cycle_node is not None:
        cur_node.next_node = cycle_node

    return root_node, get_node


def print_list(root_node):
    while root_node is not None:
        print(f"Current element is ==> {root_node.data}")
        root_node = root_node.next_node


def delete_node(del_node):
    succ = del_node.next_node
    del_node.data = succ.data
    del_node.next_node = succ.next_node


if __name__ == '__main__':
    lst, del_node = generate_list(1, 10, lambda x: x, get_at=4)
    delete_node(del_node)
    print_list(lst)

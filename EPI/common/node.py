__all__ = ['Node', 'generate_list', 'print_list']


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


if __name__ == "__main__":
    pass

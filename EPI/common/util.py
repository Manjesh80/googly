from .node import *

__all__ = ['generate_list', 'print_list', 'move_list_by', 'nudge', 'del_node']


def generate_list(start, end, func=lambda x: x):
    end += 1
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


def move_list_by(ml, n):
    while n > 0:
        n = n - 1
        ml = ml.next_node
    return ml


def nudge(ml, n=1):
    while n > 0:
        n = n - 1
        ml = ml.next_node
    return ml


def del_node(m):
    temp = m.next_node
    m.value = temp.data
    m.next_node = temp.next_node


def print_list(root_node):
    print('**********')
    while root_node is not None:
        print(f"Current element is ==> {root_node.data}")
        root_node = root_node.next_node
    print('**********')

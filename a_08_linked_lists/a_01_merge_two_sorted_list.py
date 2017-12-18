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


l1 = generate_list(1, 5, lambda x: x ** 2)
l2 = generate_list(2, 5, lambda x: x ** 3)

print_list(l1)
print_list(l2)

merged_root_node = Node(0, None)
merged_cur_node = merged_root_node

while l1 is not None and l2 is not None:
    if l1.data <= l2.data:
        merged_cur_node.next_node = l1
        l1 = l1.next_node
    else:
        merged_cur_node.next_node = l2
        l2 = l2.next_node
    merged_cur_node = merged_cur_node.next_node

merged_cur_node.next_node = l1 if l1 is not None else l2

print_list(merged_root_node.next_node)

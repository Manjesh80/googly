from ..common import *

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

if __name__ == "__main__":
    merged_cur_node.next_node = l1 if l1 is not None else l2
    print_list(merged_root_node.next_node)

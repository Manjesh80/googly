class BTNode():
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.depth = -1


def print_in_order(bt_node):
    start_node = False
    if bt_node.left.left:
        print_in_order(bt_node.left)
    else:
        start_node = True

    if start_node:
        print(f" == {bt_node.left.data or None} == ")

    print(f" == {bt_node.data or None} == ")
    print_in_order(bt_node.right)


def build_tree():
    nodes = {c: BTNode(c) for c in "abcdefghijklmnop"}

    # depth 0
    nodes['a'].left = nodes['b']
    nodes['a'].right = nodes['i']

    # depth 1
    nodes['b'].left = nodes['c']
    nodes['b'].right = nodes['f']
    nodes['i'].left = nodes['j']
    nodes['i'].right = nodes['o']

    # depth 2
    nodes['c'].left = nodes['d']
    nodes['c'].right = nodes['e']
    nodes['f'].right = nodes['g']
    nodes['j'].right = nodes['k']
    nodes['o'].right = nodes['p']

    # depth 3
    nodes['g'].left = nodes['h']
    nodes['k'].left = nodes['l']
    nodes['k'].right = nodes['n']

    # depth 4
    nodes['l'].right = nodes['m']

    # depth 5
    # nodes['m'].right = nodes['q']
    return nodes['a']


def print_in_order_traversal(root):
    print_in_order(root.left)
    print(f" == {root.data} == ")
    print_in_order(root.right)


def print_in_order(root):
    if root.left and root.left.left:
        print_in_order(root.left)
    elif root.left and not root.left.left:
        print(f" <==> {root.left.data} <==>")

    print(f" <==> {root.data} <==>")

    if root.right and root.right.right:
        print_in_order(root.right)
    elif root.right and not root.right.right:
        print(f" <==> {root.right.data} <==>")


def print_in_order_new(root):
    if root.left:
        print_in_order_new(root.left)

    print(f" <==> {root.data} <==>")

    if root.right:
        print_in_order_new(root.right)


def print_pre_order(root):
    print(f" <==> {root.data} == {root.depth} <==>")

    if root.left:
        print_pre_order(root.left)

    if root.right:
        print_pre_order(root.right)


def print_post_order(root):
    if root:
        print_post_order(root.left)
        print_post_order(root.right)
        print(f" <==> {root.data} <==>")


# 9.1 Test if a binary tree is height balanced
def calculate_tree_depth(root):
    if root:
        cur_depth = max(calculate_tree_depth(root.left),
                        calculate_tree_depth(root.right))
        root.depth = cur_depth + 1
        return root.depth
    return -1


# 9.1 Test if a binary tree is height balanced
def compute_tree_depth_from_top(root, parent_depth):
    if root:
        current_depth = parent_depth + 1
        root.depth = current_depth
        compute_tree_depth_from_top(root.left, parent_depth + 1)
        compute_tree_depth_from_top(root.right, parent_depth + 1)


if __name__ == "__main__":
    # calculate_tree_depth(nodes['a'])
    # height_of_left_tree = calculate_tree_depth(nodes['b'])
    # height_of_right_tree = calculate_tree_depth(nodes['i'])
    # print(f" Height of left left tree {height_of_left_tree+1}")
    # print(f" Height of left right tree {height_of_right_tree+1}")
    # print_pre_order(nodes['a'])
    root = build_tree()
    compute_tree_depth_from_top(root, -1)
    print_pre_order(root)

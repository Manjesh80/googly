class BTNode():
    def __init__(self, *, data, name, left=None, right=None):
        self.data = data
        self.name = data
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


def build_symmetric_tree():
    nodes = {c.split(',')[0]: BTNode(name=c.split(',')[0], data=c.split(',')[1]) for c in
             "a,314#b,6#c,2#d,3#e,6#f,2#g,3".split('#')}

    nodes['a'].left = nodes['b']
    nodes['a'].right = nodes['e']

    nodes['b'].right = nodes['c']
    nodes['c'].right = nodes['d']

    nodes['e'].left = nodes['f']
    nodes['f'].left = nodes['g']

    return nodes['a']


def build_tree():
    nodes = {c: BTNode(name=c, data=c) for c in "abcdefghijklmnopq"}

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

    # nodes['g'].right = nodes['q']
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


def depth_of_complete_sub_tree(parent_node):
    def tree_depth_from_top(root, parent_depth):
        if root:
            current_depth = parent_depth + 1
            max_depth[0] = current_depth
            root.depth = current_depth
            if root.left and root.right:
                tree_depth_from_top(root.left, current_depth)
                tree_depth_from_top(root.right, current_depth)

    max_depth = [0]
    tree_depth_from_top(parent_node, -1)
    return max_depth[0] + 1


# 9.2 Check if a tree is Symmetric
def is_tree_symmetric(root):
    def check_symmetry(left_tree, right_tree):
        if not left_tree and not right_tree:
            return True
        elif (not left_tree) ^ (not right_tree):
            return False
        elif left_tree and right_tree:
            return ((left_tree.data == right_tree.data)
                    and check_symmetry(left_tree.right, right_tree.left)
                    and check_symmetry(left_tree.left, right_tree.right))

    return check_symmetry(root.left, root.right)


if __name__ == "__main__":
    root = build_symmetric_tree()
    print(f"Is Symmetric tree ? {is_tree_symmetric(root)}")

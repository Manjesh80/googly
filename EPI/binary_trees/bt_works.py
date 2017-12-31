from collections import namedtuple


class BTNode():
    def __init__(self, *, data, name, left=None, right=None, parent=None):
        self.data = data
        self.name = name
        self.left = left
        self.right = right
        self.depth = -1
        self.parent = parent

    # def __str__(self):
    #     return str.format(" Name ==> " + self.name +
    #                       " Data ==> " + self.data)


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


def build_tree_dict():
    nodes = {c.split(',')[0]: BTNode(name=c.split(',')[0], data=c.split(',')[1]) for c in
             "a,1#b,0#c,0#d,0#e,1#f,1#g,1#h,0#i,1#j,0#k,0#l,0#m,1#n,0#o,0#p,0".split('#')}

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
    return nodes


def build_tree_dict_with_parent():
    nodes = build_tree_dict();
    nodes['b'].parent = nodes['a']
    nodes['c'].parent = nodes['b']
    nodes['d'].parent = nodes['c']
    nodes['e'].parent = nodes['c']
    nodes['f'].parent = nodes['b']
    nodes['g'].parent = nodes['f']
    nodes['h'].parent = nodes['g']
    nodes['i'].parent = nodes['a']
    nodes['j'].parent = nodes['i']
    nodes['k'].parent = nodes['j']
    nodes['n'].parent = nodes['k']
    nodes['l'].parent = nodes['k']
    nodes['m'].parent = nodes['l']
    nodes['o'].parent = nodes['i']
    nodes['p'].parent = nodes['o']
    return nodes


def build_tree():
    build_tree_dict()['a']


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


# 9.3 Compute the LCA ( lowest common ancestor ) in Binary Tree
def lca(tree, node0, node1):
    Status = namedtuple("Status", ('num_target_nodes', 'ancestor'))

    def lca_helper(tree, node0, node1):
        if not tree:
            return Status(0, None)

        left_result = lca_helper(tree.left, node0, node1)
        if left_result.num_target_nodes == 2:
            return left_result

        right_result = lca_helper(tree.right, node0, node1)
        if right_result.num_target_nodes == 2:
            return right_result

        num_target_nodes = (left_result.num_target_nodes + right_result.num_target_nodes +
                            int(tree in (node0, node1)))
        return Status(num_target_nodes, tree if num_target_nodes == 2 else None)

    return lca_helper(tree, node0, node1).ancestor


# 9.4 Compute the LCA with Parent( lowest common ancestor ) in Binary Tree
def lca_with_parent(node0, node1):
    def get_node_depth(node):
        depth = -1
        while node:
            node = node.parent
            depth += 1
        return depth

    def ascend_node(node, height):
        while height > 0:
            height -= 1
            node = node.parent
        return node

    node0_depth, node1_depth = get_node_depth(node0), get_node_depth(node1)
    depth_difference = abs(node0_depth - node1_depth)

    if node0_depth > node1_depth:
        node0 = ascend_node(node0, depth_difference)
        node0_depth -= depth_difference
    elif node1_depth > node0_depth:
        node1 = ascend_node(node1, depth_difference)
        node1_depth -= depth_difference

    while node0 is not node1:
        node0, node1 = node0.parent, node1.parent

    return node0


# 9.5 Sum the root to leaf paths
def sum_the_root_to_leaf_path(root):
    def get_child_nodes(node):
        if node:
            left_dict_values = get_child_nodes(node.left)
            right_dict_values = get_child_nodes(node.right)

            if left_dict_values or right_dict_values:
                # Iterate and add value
                merged_dict = {**left_dict_values, **right_dict_values}
                for key in merged_dict.keys():
                    merged_dict[key] = merged_dict[key].extend(node.data)
                return merged_dict
            else:
                return {node.name: [node.data]}
        else:
            return {}

    res = get_child_nodes(root)
    return res


if __name__ == "__main__":
    tree = build_tree_dict()
    parent = sum_the_root_to_leaf_path(tree['b'])
    print(parent)

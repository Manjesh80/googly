from collections import namedtuple


class BTNode:
    def __init__(self, *, data, name, left=None, right=None, parent=None, kids=-1, exploded=False):
        self.data = data
        self.name = name
        self.left = left
        self.right = right
        self.depth = -1
        self.parent = parent
        self.exploded = exploded
        self.kids = kids

    def is_leaf(self):
        return not self.left and not self.right

    def has_parent(self):
        return self.parent is not None

    def is_left_child(self):
        return self.parent and self.parent.left and self.parent.left == self

    def is_right_child(self):
        return self.parent and self.parent.right and self.parent.right == self


class Queue:
    def __init__(self, load=[]):
        self.data = load

    def enqueue(self, value):
        self.data.append(value)

    def empty(self):
        return len(self.data) == 0

    def not_empty(self):
        return len(self.data) > 0

    def dequeue(self):
        res = self.data[0]
        del self.data[0]
        return res


class Stack():
    def __init__(self):
        self.data = []

    def push(self, value):
        self.data.append(value)

    def pop(self):
        return self.data.pop()

    def peek(self):
        return self.data[-1]

    def has_item(self):
        return len(self.data) > 0


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
    nodes = {c.split(',')[0]: BTNode(name=c.split(',')[0], data=int(c.split(',')[1])) for c in
             "a,1#b,0#c,0#d,0#e,1#f,1#g,1#h,0#i,1#j,0#k,0#l,0#m,1#n,0#o,0#p,0".split('#')}

    nodes = {c.split(',')[0]: BTNode(name=c.split(',')[0], data=int(c.split(',')[1])) for c in
             "a,314#b,6#c,271#d,28#e,0#f,561#g,3#h,17#i,6#j,2#k,1#l,401#m,641#n,257#o,271#p,28".split('#')}

    nodes['a'].kids = 15
    nodes['b'].kids = 6
    nodes['c'].kids = 2
    nodes['d'].kids = 0
    nodes['e'].kids = 0
    nodes['f'].kids = 2
    nodes['g'].kids = 1
    nodes['h'].kids = 0
    nodes['i'].kids = 7
    nodes['j'].kids = 4
    nodes['k'].kids = 3
    nodes['n'].kids = 0
    nodes['l'].kids = 1
    nodes['m'].kids = 0
    nodes['o'].kids = 1
    nodes['p'].kids = 0

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
    if root:
        print(f" <==> {root.name} <==>")

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

            if left_dict_values is not None or right_dict_values is not None:
                merged_dict = {}
                if left_dict_values is not None:
                    merged_dict = dict(left_dict_values)
                if right_dict_values is not None:
                    merged_dict = {**merged_dict, **right_dict_values}

                for key in merged_dict.keys():
                    if merged_dict[key]:
                        merged_dict[key].extend(node.data)
                    else:
                        merged_dict[key] = [node.data]

                return merged_dict
            else:
                return {node.name: [node.data]}
        else:
            return None

    res = get_child_nodes(root)
    return res


# 9.6 Match sum for the path
def match_sum(root, sum_no):
    def calculate_sum(node, partial_sum):
        if not node:
            return 0

        partial_sum += node.data
        if partial_sum == sum_no:
            node_names.append(node.name)
            return partial_sum
        calculate_sum(node.left, partial_sum)
        calculate_sum(node.right, partial_sum)

    node_names = []
    calculate_sum(root, 0)
    return node_names


def sum_root_to_leaf_path(node, partial_sum=0):
    if not node:
        return 0

    partial_sum = 2 * partial_sum + node.data
    if not node.left and not node.right:
        return partial_sum

    return (sum_root_to_leaf_path(node.left, partial_sum) +
            sum_root_to_leaf_path(node.right, partial_sum))


# 9.7 Implement an inorder traversal without recursion
def traverse_in_order_stack(root):
    stack = Stack()
    stack.push(root)
    result = []
    while stack.has_item():
        ele: BTNode = stack.pop()
        if not ele.exploded and not ele.is_leaf():
            if ele.right:
                stack.push(ele.right)
            ele.exploded = True
            stack.push(ele)
            if ele.left:
                stack.push(ele.left)
            continue

        result.append(ele.name)
    return result


def traverse_in_order_two_stack(root):
    processing_stack, result = [], []

    while processing_stack or root:
        if root:
            processing_stack.append(root)
            root = root.left
        else:
            item = processing_stack.pop()
            result.append(item)
            root = item.right
    return result


# 9.8 Implement an inorder traversal without recursion
def traverse_pre_order_stack(root):
    stack = Stack()
    stack.push(root)
    result = []
    while stack.has_item():
        ele: BTNode = stack.pop()
        if not ele.exploded and not ele.is_leaf():
            if ele.right:
                stack.push(ele.right)
            if ele.left:
                stack.push(ele.left)
            ele.exploded = True
            stack.push(ele)
            continue

        result.append(ele.name)
    return result


def traverse_pre_order_two_stack(root):
    processing_stack, result = [], []
    while processing_stack or root:
        if root:
            result.append(root.name)
            processing_stack.append(root.right)
            root = root.left
        else:
            root = processing_stack.pop()
    return result


# 9.9 Compute the Kth node in In order traversal
# [ D , C , E , B, F, H, G , A , J , L, M , K , N , I, O , P]
# k = 6 ==> H
def compute_kth_node_in_order_travresal(root, k):
    def travel(node, n):
        if not node:
            return None
        left_count = node.left.kids + 1 if node.left else 0
        current_node_index = left_count + 1

        if n == current_node_index:
            return node
        elif n <= left_count:
            return travel(node.left, n)
        elif n > left_count + 1:
            return travel(node.right, n - (left_count + 1))

    if k > root.kids + 1:
        raise AttributeError("K should be lesser than root count")
    res = travel(root, k)
    return res


def find_succ(root, match):
    def find_match(a, b):

        if a:
            return a if a is b else None

    def find_successor(root, match):
        while root:
            if root:
                found = find_match(root.left, match)
                if found:
                    return found
                if root is match:
                    return root.right
                return find_successor(root.right, match)
            else:
                return None

    return find_successor(root, match)


# 9.10 Compute the successor
def find_successor(root, match):
    def process_in_order(root):
        if root:
            if root.left and root.left is not match and len(match_added) == 1 and len(successor) == 0:
                successor.append(root.left)
                return
            process_in_order(root.left)
            if root is match:
                match_added.append(True)
            if root is not match and len(match_added) == 1 and len(successor) == 0:
                successor.append(root)
                return
            process_in_order(root.right)

    successor = []
    match_added = []
    process_in_order(root)
    return successor[0]


def find_successor_epi(node):
    if node.right:
        node = node.right
        while node.left:
            node = node.left
        return node

    while node.parent and node.parent.right is node:
        node = node.parent

    return node.parent


# 9.11 Traverse in order
def traverse_in_order(tree: BTNode):
    prev: BTNode = None
    result = []

    while tree:
        if prev is tree.parent:
            if tree.left:
                next = tree.left
            else:
                result.append(tree.name)
                next = tree.right or tree.parent
        elif tree.left is prev:
            result.append(tree.name)
            next = tree.right or tree.parent
        else:
            next = tree.parent

        prev, tree = tree, next

    return result


# 9.12 Reconstruct a binary tree from traversal data
# inorder  ==> {  F, B, A , E, H , C , D , I , G }
# preorder ==> { H , B, F, E , A , C , D , G, I }
def build_binary_tree(preorder, inorder):
    def build_binary_tree_helper(pre_order_start, pre_order_end, inorder_start, inorder_end):
        if pre_order_start > pre_order_end or inorder_start > inorder_end:
            return None
        left_tree_size = inorder.index(preorder[pre_order_start]) + 1

        return BTNode(data=preorder[pre_order_start], name=preorder[pre_order_start],
                      left=build_binary_tree_helper(pre_order_start + 1, left_tree_size - 1, pre_order_start,
                                                    left_tree_size),
                      right=build_binary_tree_helper(left_tree_size + 1, pre_order_end, left_tree_size + 1, inorder_end)
                      )

    return build_binary_tree_helper(0, len(preorder) - 1, 0, len(inorder) - 1)


def build_binary_tree_epi(preorder, inorder):
    node_to_inorder_idx = {data: i for i, data in enumerate(inorder)}

    def build_binary_tree_helper(pre_order_start, pre_order_end, inorder_start, inorder_end):
        if pre_order_start >= pre_order_end or inorder_start >= inorder_end:
            return None

        root_inorder_index = node_to_inorder_idx[preorder[pre_order_start]]
        left_tree_size = root_inorder_index - inorder_start

        return BTNode(data=preorder[pre_order_start], name=preorder[pre_order_start],
                      left=build_binary_tree_helper(pre_order_start + 1, pre_order_start + 1 + left_tree_size,
                                                    inorder_start, root_inorder_index),
                      right=build_binary_tree_helper(pre_order_start + 1 + left_tree_size, pre_order_end,
                                                     root_inorder_index + 1, inorder_end))

    return build_binary_tree_helper(0, len(preorder), 0, len(inorder))


# 9.13 Reconstruct a binary tree with markers
def build_pre_order_tree(q: Queue):
    if q.not_empty():
        value = q.dequeue()
        if value:
            parent = BTNode(name=value, data=value,
                            left=build_pre_order_tree(q),
                            right=build_pre_order_tree(q))
            return parent


# 9.15 build leaves and edges
def build_leaves_and_edges(root):
    def traverse_in_order_and_get_leaves(node: BTNode):
        if node:
            traverse_in_order_and_get_leaves(node.left)
            if node.is_leaf():
                result.append(node)
            traverse_in_order_and_get_leaves(node.right)

    def left_traverse(node):
        if node:
            result.append(node)
            if node.right:
                traverse_in_order_and_get_leaves(node.right)
            left_traverse(node.left)

    def right_traverse(node):
        if node:
            result.append(node)
            if node.left:
                traverse_in_order_and_get_leaves(node.left)
            right_traverse(node.right)

    def traverse_and_add(node):
        if node:
            result.append(node)
            left_traverse(node.left)
            right_traverse(node.right)

    result = []
    traverse_and_add(root)
    return result


if __name__ == "__main__":
    nodes = build_tree_dict_with_parent()
    res = build_leaves_and_edges(nodes['a'])
    [print(x.name) for x in res]

#
#
#
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top
# comment to keep code on top


# comment to keep code on top

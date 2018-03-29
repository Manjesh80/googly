from collections import namedtuple


class BTNode:
    def __init__(self, *, data, name, left=None, right=None, parent=None, kids=-1, exploded=False, rlink=None,
                 is_locked=False):
        self.data = data
        self.name = name
        self.left = left
        self.right = right
        self.depth = -1
        self.parent = parent
        self.exploded = exploded
        self.kids = kids
        self.rlink = rlink
        self.is_locked = is_locked

    def is_locked(self):
        return self.is_locked

    def acquire_lock(self):
        if not self.is_locked:
            self.is_locked = True
            return True
        else:
            return False

    def release_lock(self):
        if self.is_locked:
            self.is_locked = False
            return True
        else:
            return False

    def is_leaf(self):
        return not self.left and not self.right

    def is_not_leaf_and_balanced(self):
        return self.left and self.right

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


def build_perfect_tree():
    nodes = {c.split(',')[0]: BTNode(name=c.split(',')[0], data=c.split(',')[0]) for c in
             "a#b#c#d#e#f#g#h#i#j#k#l#m#n#o".split('#')}

    nodes['a'].left = nodes['b']
    nodes['a'].right = nodes['i']
    nodes['b'].left = nodes['c']
    nodes['b'].right = nodes['f']
    nodes['c'].left = nodes['d']
    nodes['c'].right = nodes['e']
    nodes['f'].left = nodes['g']
    nodes['f'].right = nodes['h']
    nodes['i'].left = nodes['j']
    nodes['i'].right = nodes['m']
    nodes['j'].left = nodes['k']
    nodes['j'].right = nodes['l']
    nodes['m'].left = nodes['n']
    nodes['m'].right = nodes['o']

    # nodes['c'].parent = nodes['b']
    # nodes['d'].parent = nodes['c']
    # nodes['e'].parent = nodes['c']
    # nodes['f'].parent = nodes['b']
    # nodes['g'].parent = nodes['f']
    # nodes['h'].parent = nodes['f']

    return nodes


def build_perfect_tree_new():
    nodes = {c.split(',')[0]: BTNode(name=c.split(',')[0], data=c.split(',')[0]) for c in
             "a#b#c#d#e#f#g#h#i".split('#')}

    nodes['a'].left = nodes['b']
    nodes['a'].right = nodes['d']
    nodes['b'].left = nodes['c']
    nodes['d'].left = nodes['e']
    nodes['d'].right = nodes['g']
    nodes['e'].left = nodes['f']
    nodes['g'].left = nodes['h']
    nodes['g'].right = nodes['i']

    return nodes


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


def print_in_order_traversal(root: BTNode):
    if not root:
        return
    print_in_order_traversal(root.left)
    print(f" == {root.name} ")
    print_in_order_traversal(root.right)


def print_in_order_traversal_with_rlink(root):
    if root:
        print_in_order_traversal_with_rlink(root.left)
        if root.rlink:
            print(f" == {root.data} has RLNIK = {root.rlink.data}")
        print_in_order_traversal_with_rlink(root.right)


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

    print(f" <==> {root.name} <==>")

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


def is_height_balanced(root):
    BH = namedtuple('BH', ('is_balanced', 'height'))

    def calculate_hd(node):
        if not node:
            return BH(is_balanced=True, height=-1)

        left_bh = calculate_hd(node.left)
        if not left_bh.is_balanced:
            return BH(is_balanced=False, height=0)

        right_bh = calculate_hd(node.right)
        if not right_bh.is_balanced:
            return BH(is_balanced=False, height=0)

        balanced = abs(left_bh.height - right_bh.height) <= 1
        height = max(left_bh.height, right_bh.height) + 1
        return BH(is_balanced=balanced, height=height)

    return calculate_hd(root)


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


def compute_lca(root, node1, node2):
    def do_post_order(node):
        if not node:
            return 0
        left_added = do_post_order(node.left)
        right_added = do_post_order(node.right)
        # if len(found_so_far) == 2 and parent[0] is None and (left_added + right_added) == 2:
        if (left_added + right_added) == 2:
            parent[0] = node
            return 0
        if node.name in [node1.name, node2.name]:
            found_so_far.append(node.name)
            return 1
        else:
            return left_added + right_added

    parent = [None]
    found_so_far = []
    do_post_order(root)
    print(f"Parent is ==> {parent[0].name}")
    return parent[0]


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


def sum_the_root_to_leaf_path_new(root):
    LevelWithSum = namedtuple('LevelWithSum', ('left_level', 'left_sum', 'right_level', 'right_sum'))

    def traverse(node):
        if not node:
            return LevelWithSum(-1, 0, -1, 0)
        left_value = traverse(node.left)
        right_value = traverse(node.right)
        value = LevelWithSum(-1, 0, -1, 0)
        if left_value:
            value.left_level = left_value.left_level + 1
            value.left_sum = (node.data * (2 ** value.left_level)) + left_value.left_sum
        if right_value:
            value.right_level = right_value.right_level + 1
            value.right_sum = (node.data * (2 ** value.right_level)) + right_value.right_sum
        return value

    return traverse(root)


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


def compute_kth_node_in_order_traversal_new(root, k):
    def in_order(node):
        if not node or current_number[0] > k:
            return
        in_order(node.left)
        print(f"Processing node ==> {node.name}")
        current_number[0] += 1
        if current_number[0] == k:
            print(f"k the element found ==> {node.name}")
            k_th_element[0] = node
        in_order(node.right)

    current_number = [0]
    k_th_element = [None]
    in_order(root)
    return k_th_element[0]


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


def find_successor_new(root, predecessor):
    def in_order(node):
        if not node or successor[0] is not None:
            return
        in_order(node.left)
        if predecessor_found[0] and successor[0] is None:
            successor[0] = node
        if predecessor.name == node.name:
            predecessor_found[0] = True
        in_order(node.right)

    predecessor_found = [False]
    successor = [None]
    in_order(root)
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


def traverse_in_order_new(root: BTNode):
    curr = root
    added_by_left_child = False
    while curr:
        if not added_by_left_child:
            if curr.left:
                curr = curr.left
                continue
            else:
                print(f"Processing left leave element ==> {curr.name}")
                curr = curr.parent
                added_by_left_child = True

        print(f"Processing parent element ==> {curr.name}")

        if curr.right:
            curr = curr.right
            continue


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


def build_binary_tree_new(preorder, inorder):
    def build_tree(po, io):
        if not po and not io:
            return None
        return BTNode(name=po[0], data=po[0],
                      left=build_tree(po=po[1:io.index(po[0]) + 1], io=io[0:io.index(po[0])]),
                      right=build_tree(po=po[io.index(po[0]) + 1:], io=io[io.index(po[0]) + 1:]))

    return build_tree(preorder, inorder)


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


def build_pre_order_tree_new(values):
    def pre_order(values):
        if values:
            curr = values.pop(0)
            if curr:
                return BTNode(name=curr, data=curr,
                              left=pre_order(values),
                              right=pre_order(values))


# 9.14 Build linked list from edges
def build_linked_list_from_edges(root):
    def in_order(node: BTNode):
        if not node:
            return
        in_order(node.left)
        if node.is_leaf():
            result.append(node.name)
        in_order(node.right)

    result = []
    in_order(root)
    return result


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
            left_traverse(node.left)
            if node.right:
                traverse_in_order_and_get_leaves(node.right)

    def right_traverse(node):
        if node:
            result.append(node)
            right_traverse(node.right)
            if node.left:
                traverse_in_order_and_get_leaves(node.left)

    def traverse_and_add(node):
        if node:
            result.append(node)
            left_traverse(node.left)
            right_traverse(node.right)

    result = []
    traverse_and_add(root)
    return result


# 9.16 Fill RLink
def build_rlink(root):
    def emit_right_nodes(node):
        result = []
        if node:
            while node:
                result.append(node)
                node = node.right
        return result

    def emit_left_nodes(node):
        result = []
        if node:
            while node:
                result.append(node)
                node = node.left
        return result

    def marry_right_to_left(righties, lefties):
        for rightie, leftie in zip(righties, lefties):
            rightie.rlink = leftie

    def build_rlink_inorder(root: BTNode):
        if root:
            left_node = build_rlink_inorder(root.left)
            right_node = build_rlink_inorder(root.right)

            righties = emit_right_nodes(left_node)
            lefties = emit_left_nodes(right_node)

            if righties and lefties:
                marry_right_to_left(righties, lefties)
            return root

    return build_rlink_inorder(root)


# 9.16 Fill RLink
def build_rlink_new(root):
    def in_order(node):
        if not node:
            return
        if node.left and node.right:
            print(f" Marry in-order nodes ==> {node.left.name} ==> {node.right.name}")
        in_order(node.left)
        in_order(node.right)

    def marry_right_wall_to_left_wall(leftie, rightie):
        if not leftie or not rightie:
            return
        left_wall = leftie.right or leftie.left or None
        right_wall = rightie.left or rightie.right or None
        if left_wall and right_wall:
            print(f" Marry cross-over nodes ==> {left_wall.name} ==> {right_wall.name}")
            marry_right_wall_to_left_wall(left_wall, right_wall)

        marry_right_wall_to_left_wall(leftie.left, leftie.right)
        marry_right_wall_to_left_wall(rightie.left, rightie.right)

    in_order(root)
    marry_right_wall_to_left_wall(root.left, root.right)


def construct_right_sibling(tree):
    def populate_children_next_filed(start_node):
        while start_node and start_node.left:
            start_node.left.rlink = start_node.right
            start_node.right.rlink = start_node.rlink and start_node.rlink.left
            start_node = start_node.rlink

    while tree and tree.left:
        populate_children_next_filed(tree)
        tree = tree.left


# 9.17 Build lock API
def any_child_lock(node: BTNode):
    if node:
        left_state = any_child_lock(node.left)
        right_state = any_child_lock(node.right)
        return node.is_locked or left_state or right_state


def any_parent_lock(node):
    result = False
    while node:
        result = result or node.is_locked
        node = node.parent
    return result


def lock_node(node):
    if not any_child_lock(node) and not any_parent_lock(node):
        res = node.acquire_lock()
        if res:
            return res
        else:
            raise ValueError("Cannot get lock")
    else:
        return False


def unlock_node(node):
    return node.release_lock()


# TODO Reimplement 9.11 and 9.12
# 9.12 Reconstruct a binary tree from traversal data
# inorder  ==> {  F, B, A , E, H , C , D , I , G }
# preorder ==> { H , B, F, E , A , C , D , G, I }
def build_binary_tree_ext(preorder, inorder):
    def build_node(anchor, por, ior):
        if anchor:
            anchor_ior_index = ior.index(anchor) if ior and ior.count(anchor) > 0 else -1
            anchor_por_index = por.index(anchor) if por and por.count(anchor) > 0 else -1

            lst = ior[0:anchor_ior_index]
            rst = ior[(anchor_ior_index + 1): len(ior)]

            node = BTNode(data=anchor, name=anchor)

            if len(lst) > 0:
                left_anchor = por[anchor_por_index + 1]
                left_por = por[anchor_por_index:anchor_ior_index + 1]
                left_ior = ior[0:anchor_ior_index]
                left_node = build_node(left_anchor, left_por, left_ior)
                node.left = left_node

            if len(rst) > 0:
                right_anchor = por[anchor_por_index + 1 + len(lst)]
                right_por = por[anchor_ior_index + 1:len(por)]
                right_ior = ior[anchor_ior_index + 1:len(ior)]
                right_node = build_node(right_anchor, right_por, right_ior)
                node.right = right_node

            return node

    return build_node(preorder[0], preorder, inorder)


if __name__ == "__main__":
    # nodes = build_tree_dict()
    # print_in_order_traversal(nodes['a'])
    # res = build_linked_list_from_edges(nodes['a'])
    # print(res)

    nodes = build_perfect_tree_new()
    build_rlink_new(nodes['a'])

    # res = compute_kth_node_in_order_traversal_new(nodes['a'], 6)
    # if res:
    #     print(res.name)
    # else:
    #     print(None)

    # preorder = ['H', 'B', 'F', 'E', 'A', 'C', 'D', 'G', 'I']
    # inorder = ['F', 'B', 'A', 'E', 'H', 'C', 'D', 'I', 'G']
    # res = build_binary_tree_new(preorder, inorder)
    # print(res)
    # print_in_order_traversal(res)
    # # preorder = ['H', 'B', 'F', 'E', 'A']
    # # inorder = ['F', 'B', 'A', 'E', 'H']
    # node = build_binary_tree_ext(preorder, inorder)
    # print_in_order_new(node)

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

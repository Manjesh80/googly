from bintrees import *
from bintrees import abctree
import bintrees


class BSTNode:
    def __init__(self, *, name=None, data=None, left=None, right=None):
        self.name, self.data, self.left, self.right = name, data, left, right

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_not_leaf(self):
        return not self.is_leaf()

    def is_left_child(self, parent):
        return self is parent.left


def build_number_tree_nodes():
    A_DICT = {i: BSTNode(name=i, data=i) for i in range(1, 16)}

    A_DICT[8].left = A_DICT[4]
    A_DICT[8].right = A_DICT[12]

    return A_DICT


def build_number_tree_node():
    return build_number_tree_nodes()[8]


def build_tree_nodes():
    A = "A,19#B,7#C,3#D,2#E,5#F,11#G,17#H,13#I,43#J,23#K,37#L,29#M,31#N,41#O,47#P,53"
    A_DICT = {i.split(',')[0]: BSTNode(name=i.split(',')[0], data=int(i.split(',')[1]))
              for i in A.split('#')}
    A_DICT['A'].left = A_DICT['B']
    A_DICT['A'].right = A_DICT['I']

    A_DICT['B'].left = A_DICT['C']
    A_DICT['B'].right = A_DICT['F']

    A_DICT['C'].left = A_DICT['D']
    A_DICT['C'].right = A_DICT['E']

    A_DICT['F'].right = A_DICT['G']
    A_DICT['G'].left = A_DICT['H']

    A_DICT['I'].left = A_DICT['J']
    A_DICT['I'].right = A_DICT['O']

    A_DICT['J'].right = A_DICT['K']

    A_DICT['K'].left = A_DICT['L']
    A_DICT['K'].right = A_DICT['N']

    A_DICT['L'].right = A_DICT['M']

    A_DICT['O'].right = A_DICT['P']
    return A_DICT


def build_tree():
    return build_tree_nodes()['A']


def in_order_traversal(root):
    if root:
        in_order_traversal(root.left)
        print(f" ********* {root.data} ********* ")
        in_order_traversal(root.right)


def in_order_traversal_emit(root):
    def in_order_traversal(root):
        if root:
            in_order_traversal(root.left)
            result.append(root.name)
            in_order_traversal(root.right)

    result = []
    in_order_traversal(root)
    return result


def post_order_traversal_emit(root):
    def post_order_traversal(root):
        if root:
            post_order_traversal(root.left)
            post_order_traversal(root.right)
            result.append(root.name)

    result = []
    post_order_traversal(root)
    return result


def pre_order_traversal_emit(root):
    def pre_order_traversal(root):
        if root:
            result.append(root.name)
            pre_order_traversal(root.left)
            pre_order_traversal(root.right)

    result = []
    pre_order_traversal(root)
    return result


def post_order_traversal(root):
    if root:
        post_order_traversal(root.left)
        post_order_traversal(root.right)
        print(f" ********* {root.name} ********* ")


def pre_order_traversal(root):
    def traversal(root):
        if root:
            result.append(root.data)
            print(root.data)
            traversal(root.left)
            traversal(root.right)

    result = []
    traversal(root)
    return result


def search_bst(root, value):
    def search(root):
        if root:
            if value == root.data:
                return value
            elif value < root.data:
                return search(root.left)
            elif value > root.data:
                return search(root.right)
            else:
                return None

    return search(root)


# 14.1 Is tree a BST
def is_tree_bst(root):
    def is_sub_tree_bst(root):
        if root:
            is_sub_tree_bst(root.left)
            if result[0]:
                parent_value = root.data
                left_value = root.left.data if root.left else float('-inf')
                right_value = root.right.data if root.right else float('inf')
                result[0] = result[0] and (left_value <= parent_value <= right_value)

    result = [True]
    is_sub_tree_bst(root)
    return result[0]


def is_tree_bst_new(root):
    def is_sub_tree_bst(root):
        if root:
            res = is_sub_tree_bst(root.left)
            if res:
                parent_value = root.data
                left_value = root.left.data if root.left else float('-inf')
                right_value = root.right.data if root.right else float('inf')
                res = left_value <= parent_value <= right_value
            return res
        else:
            return True

    return is_sub_tree_bst(root)


# TODO Page 205 , Implement breadth first search

# 14.2 Search next highest
def search_next_highest(root, value):
    def find_min_in_bst(root):
        return 100

    def search(root):
        if root:
            if value == root.data and not found[0]:
                found[0] = True
            elif value < root.data and not found[0]:
                search(root.left)
            elif value > root.data and not found[0]:
                search(root.right)

            if found[0]:
                return root

            if found[0] and not root.right:
                return None

            if found[0] and root.right:
                return find_min_in_bst(root.right)

    found = [False]
    return search(root)


# 14.2 Search next highest
def search_and_next_epi(root, value):
    sub_root, first_so_far = root, None
    while sub_root:
        if sub_root.data > value:
            first_so_far, sub_root = sub_root, sub_root.left
        else:
            sub_root = sub_root.right

    return first_so_far


def demo_rb_tree():
    t: abctree = RBTree([(1, 1), (2, 2), (3, 3)])
    print(t.min_item())


# 14.3 Find k largest elements
def reverse_order_traversal(root, count):
    def reverse_order(root):
        if root and len(counter) < count:
            reverse_order(root.right)
            if len(counter) < count:
                counter.append(root.data)
                reverse_order(root.left)

    counter = []
    reverse_order(root)
    return counter


# 14.2 Search next highest
def search_and_next(root, value):
    def search(root):
        if root:
            found = search(root.left)
            if found and not next_element[0]:
                next_element[0] = root
                return found
            if not found:
                if root.data == value:
                    return True
                if not found[0]:
                    search(root.right)

        # if found[0] and not next_element[0]:
        #     next_element[0] = root

    found = [False]
    next_element = [None]
    search(root)
    return found[0], next_element[0]


# 14.4 Compute LCA in a BST
def compute_lca_bst(root, child1, child2):
    def is_parent_in_between(root):
        if root:
            if child1.data <= root.data <= child2.data:
                lca[0] = root
                return
            elif (child1.data <= root.data) and (child2.data <= root.data):
                is_parent_in_between(root.left)
            elif (child1.data >= root.data) and (child2.data >= root.data):
                is_parent_in_between(root.right)

    lca = [None]
    is_parent_in_between(root)
    return lca[0]


# 14.5 Reconstruct a tree from a traversal data
def reconstruct_bst_from_traversal_data(A):
    def break_list(root_value, A):
        break_index = len(A)
        for i in range(0, len(A)):
            if A[i] > root_value:
                break_index = i
                break
        return A[:break_index], A[break_index:]

    def build_bst_tree(A):
        if A:
            left_nodes, right_nodes = break_list(A[0], A[1:])
            root_node = BSTNode(name=A[0], data=A[0],
                                left=build_bst_tree(left_nodes),
                                right=build_bst_tree(right_nodes))
            return root_node

    return build_bst_tree(A)


# 14.5 Reconstruct a tree from a traversal data
def reconstruct_bst_from_traversal_data_o_n(A):
    def build_bst_tree(start, end):
        if current_index[0] < len(A):
            root_value = A[current_index[0]]
            if not start <= root_value <= end:
                return None
            current_index[0] += 1
            left_subtree = build_bst_tree(start, root_value)
            right_subtree = build_bst_tree(root_value, end)
            return BSTNode(name=root_value, data=root_value,
                           left=left_subtree, right=right_subtree)

    current_index = [0]
    return build_bst_tree(float('-inf'), float('inf'))


# 14.6 Find closest entry in three sorted arrays
def closest_entry_in_3_arrays(sorted_arrays):
    bst = bintrees.RBTree()
    for idx, sorted_array in enumerate(sorted_arrays):
        array_iter = iter(sorted_array)
        min_value = next(array_iter, None)
        if min_value is not None:
            bst.insert((min_value, idx), array_iter)

    best_min_distance_so_far = float('inf')
    while True:
        min_key, min_iter = bst.min_item()
        max_key, max_iter = bst.max_item()
        min_distance_now = max_key[0] - min_key[0]
        best_min_distance_so_far = min(best_min_distance_so_far, min_distance_now)
        bst.pop_min()
        next_value = next(min_iter, None)
        if next_value is None:
            return best_min_distance_so_far
        bst.insert((next_value, min_key[1]), min_iter)

    return best_min_distance_so_far


# 14.9 Build a balances BST
def build_a_balanced_sbt(sorted_array):
    def build_bst_tree(sorted_array):
        if sorted_array:
            mid = len(sorted_array) // 2
            root = BSTNode(name=sorted_array[mid], data=sorted_array[mid],
                           left=build_bst_tree(sorted_array[0:mid]),
                           right=build_bst_tree(sorted_array[mid + 1:]))
            return root

    return build_bst_tree(sorted_array)


# 14.10 Delete a node in BST
def delete_node_in_bst_old(root, value):
    parent, node_to_delete = search_node_with_parent(root, value)

    if node_to_delete and node_to_delete.is_leaf():
        if node_to_delete.is_left_child(parent):
            parent.left = None
        else:
            parent.right = None
        del node_to_delete

    if node_to_delete and node_to_delete.is_not_leaf():
        if node_to_delete.is_left_child(parent) and node_to_delete.right is not None:
            parent.left = node_to_delete.right
        elif node_to_delete.is_left_child(parent) and node_to_delete.right is None:
            parent.left = node_to_delete.left
        elif not node_to_delete.is_left_child(parent) and node_to_delete.right is not None:
            parent.right = node_to_delete.right
        elif not node_to_delete.is_left_child(parent) and node_to_delete.right is None:
            pass


def search_node_with_parent(root, value):
    def search(parent, sub_tree):
        if sub_tree:
            if value == sub_tree.data:
                return parent, sub_tree

            parent = sub_tree
            sub_tree = sub_tree.left if value < sub_tree.data else sub_tree.right
            return search(parent, sub_tree)

    return search(root, root)


def leaf_left_child(root):
    if root:
        while root.left:
            root = root.left
    return root


def delete_node_in_bst(root, value):
    parent, node_to_delete = search_node_with_parent(root, value)
    if node_to_delete and node_to_delete.is_not_leaf():
        # Get Lowest Subtree
        lowest_in_right_subtree = leaf_left_child(node_to_delete.right)
        # If found then set it to parents left
        if lowest_in_right_subtree and parent and parent is not node_to_delete:

            if lowest_in_right_subtree is not node_to_delete.left:
                lowest_in_right_subtree.left = node_to_delete.left
                parent.left = node_to_delete.right

            else:
                parent.right = node_to_delete.left
                node_to_delete.left.right = node_to_delete.right

        if lowest_in_right_subtree and parent is node_to_delete:
            lowest_in_right_subtree.left = node_to_delete.left
            root = node_to_delete.right

    if node_to_delete and node_to_delete.is_leaf():
        if parent is not node_to_delete:
            if node_to_delete.is_left_child(parent):
                parent.left = None
            else:
                parent.right = None
    return root


# 14.11 Test if 3 BST nodes are totally ordered
def are_3_nodes_orderes(A, B, C):
    if B in (A, C):
        return False
    result = False
    A_tree = A
    C_tree = C
    did_C_reach_middle = False
    did_A_reach_middle = False

    def advance_to_node(s, e):
        node = s
        res = False
        if s:
            if s.data > e.data:
                node = s.left
            elif s.data < e.data:
                node = s.right
            else:
                res = True
        return (node, res)

    while True:
        A_tree, did_A_reach_middle = advance_to_node(A_tree, B)
        if did_A_reach_middle:
            break
        C_tree, did_C_reach_middle = advance_to_node(C_tree, B)
        if did_C_reach_middle:
            break

    did_A_reach_end = False
    did_C_reach_end = False

    if did_A_reach_middle or did_C_reach_middle:
        while True:
            if did_A_reach_middle:
                A_tree, did_A_reach_end = advance_to_node(A_tree, C)
                if did_A_reach_end or A_tree is None:
                    break
            if did_C_reach_middle:
                C_tree, did_C_reach_end = advance_to_node(C_tree, C)
                if did_C_reach_end or C_tree is None:
                    break

    # Who ever reaches first, then go and find other
    return did_A_reach_end or did_C_reach_end


# 14.12 Range Lookup
def range_lookup(root, start, end):
    def in_order_traverse(root):
        if (root):
            in_order_traverse(root.left)
            if root.data >= start and root.data <= end:
                range.append(root.data)
            in_order_traverse(root.right)

    range = []
    in_order_traverse(root)
    return range


# 14.12 Range Lookup
def range_lookup_opt(root, start, end):
    def in_between(num, start, end):
        return num in range(start, end)

    def opt_pre_order_traversal(root):
        if root:
            if in_between(root.data, start, end):
                result.append(root.data)

            if root.left:
                if in_between(root.left.data, start, end):
                    result.append(root.left.data)

                if root.left.data >= start:
                    opt_pre_order_traversal(root.left.left)
                    opt_pre_order_traversal(root.left.right)
                else:
                    opt_pre_order_traversal(root.left.right)

            if root.right:
                if in_between(root.right.data, start, end):
                    result.append(root.right.data)

                if root.right.data <= end:
                    opt_pre_order_traversal(root.right.left)
                    opt_pre_order_traversal(root.right.right)
                else:
                    opt_pre_order_traversal(root.right.left)

    result = []
    opt_pre_order_traversal(root)
    return result


if __name__ == "__main__":
    # nodes = build_tree_nodes()
    # res = range_lookup(nodes['A'], 12, 38)

    # root = build_a_balanced_sbt([*range(1, 8)])

    # root = build_a_balanced_sbt([*range(1, 8)])
    # res = range_lookup_opt(root, 3, 6)

    root = build_a_balanced_sbt([*range(1, 32)])
    res = range_lookup_opt(root, 3, 27)

    print(res)
    res.sort()
    print(set(res))

    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top
    # Comment to move file to top

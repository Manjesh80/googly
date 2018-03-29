from bintrees import rbtree


class BSTNode:
    def __init__(self, key=None, value=None, parent=None, left=None, right=None, is_root=False):
        self.key = key
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right
        self.is_root = is_root

    def is_leaf(self):
        return not (self.left or self.right)

    def is_left_child(self):
        if self.parent:
            return True if self.parent.left is self else False
        else:
            return True

    def is_right_child(self):
        if self.parent:
            return True if self.parent.right is self else False
        else:
            return True


def build_tree_nodes():
    A = "A,19#B,7#C,3#D,2#E,5#F,11#G,17#H,13#I,43#J,23#K,37#L,29#M,31#N,41#O,47#P,53"
    A_DICT = {i.split(',')[0]: BSTNode(key=i.split(',')[0], value=int(i.split(',')[1]))
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


def search_bst(root, key):
    if not root or root.key == key:
        return root
    elif key > root.key:
        return search_bst(root.right, key)
    else:
        return search_bst(root.left, key)


def traverse_bin_tree(root):
    def is_valid_bst(node: BSTNode, is_left, parent_value, grand_parent_value):
        if not node:
            return True
        print(f"Processing node ==> {node.key}")
        if node.is_leaf() and is_left:
            return True if node.value < parent_value else False
        if node.is_leaf() and not is_left:
            return True if (node.value > parent_value) else False
        if (is_left and node.value < parent_value) or \
                (not is_left and node.value > parent_value):
            return is_valid_bst(node.left, True, node.value, parent_value) and \
                   is_valid_bst(node.right, False, node.value, parent_value)
        else:
            return False

    return is_valid_bst(root, False, float('-inf'), float('inf'))


# nodes = build_tree_nodes()
# res = find_next_greater(nodes['A'], 12)
def find_next_greater(root, check_value):
    def keep_moving(node):
        if not node:
            return
        if check_value < node.value:
            result[0] = node.value
            keep_moving(node.left)
        else:
            keep_moving(node.right)

    result = [None]
    keep_moving(root)
    return result


# nodes = build_tree_nodes()
# res = kth_highest(nodes['A'], 6)
def kth_highest(root, k):
    def right_order_traversal(node):
        if not node:
            return
        right_order_traversal(node.right)
        if result[0] < k:
            result[0] += 1
            result[1] = node.value
            print(f"Value is ==> {node.value}")
            right_order_traversal(node.left)

    result = [0, None]
    right_order_traversal(root)
    return result


def find_lca(root, child1, child2):
    child1, child2 = (child1, child2) if child1.value < child2.value else (child2, child1)

    def lca(node):
        if (not node) or (result[0] is not None): return
        if node.value > child1.value and node.value < child2.value:
            result[0] = node.key
        elif node.value > child1.value and node.value > child2.value:
            lca(node.left)
        else:
            lca(node.right)

    result = [None]
    lca(root)
    return result


def range_lookup(root, lower, upper):
    def grab_range(node):
        if not node: return
        if node.value > upper:
            grab_range(node.left)
        elif node.value < lower:
            grab_range(node.right)
        else:
            grab_range(node.left)
            result.append(node.value)
            grab_range(node.right)

    result = []
    grab_range(root)
    return result


class LLRBTree:
    def __init__(self):
        self.root = None

    def insert(self, key, value):
        pass

    def remove(self, key):
        pass

    def __left__rotate(self, node):
        pass

    def __right__rotate(self, node):
        pass

    def __flip__colours(self, node):
        pass


if __name__ == "__main__":
    nodes = build_tree_nodes()
    res = range_lookup(nodes['A'], 1, 54)
    print(res)

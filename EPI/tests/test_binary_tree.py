from ..binary_trees import *


# 9.1 Test if a binary tree is height balanced
# pytest EPI\tests\test_binary_tree.py::test_calculate_tree_depth
def test_calculate_tree_depth():
    root = build_tree()
    height_of_left_tree = calculate_tree_depth(root.left)
    height_of_right_tree = calculate_tree_depth(root.right)
    assert abs(height_of_left_tree - height_of_right_tree) < 2


# 9.2 Check if a tree is Symmetric
# pytest EPI\tests\test_binary_tree.py::test_is_tree_symmetric
def test_is_tree_symmetric():
    root = build_symmetric_tree()
    assert True == is_tree_symmetric(root)


# 9.4 Compute the LCA with Parent( lowest common ancestor ) in Binary Tree
# pytest EPI\tests\test_binary_tree.py::test_lca_with_parent
def test_lca_with_parent():
    tree = build_tree_dict_with_parent()
    parent = lca_with_parent(tree['m'], tree['p'])
    assert parent.name == 'i'


# 9.6 Match sum for the path
# pytest EPI\tests\test_binary_tree.py::sum_the_root_to_leaf_path
def test_match_sum():
    tree = build_tree_dict()
    parent = sum_the_root_to_leaf_path(tree['a'])
    parent = match_sum(tree['a'], 591)
    assert len(parent) == 2


# 9.7 Implement an inorder traversal without recursion
# pytest -s EPI\tests\test_binary_tree.py::test_traverse_in_order_stack
def test_traverse_in_order_stack():
    res = traverse_in_order_stack(build_tree_dict()['a'])
    assert len(res) == 16


# 9.10 Compute the successor
# pytest -s EPI\tests\test_binary_tree.py::test_find_successor
def test_find_successor():
    nodes = build_tree_dict()
    assert find_successor(nodes['a'], nodes['d']).name == 'c'
    assert find_successor(nodes['a'], nodes['c']).name == 'e'
    assert find_successor(nodes['a'], nodes['e']).name == 'b'
    assert find_successor(nodes['a'], nodes['b']).name == 'f'
    assert find_successor(nodes['a'], nodes['f']).name == 'h'
    assert find_successor(nodes['a'], nodes['h']).name == 'g'
    assert find_successor(nodes['a'], nodes['g']).name == 'a'
    assert find_successor(nodes['a'], nodes['a']).name == 'j'
    assert find_successor(nodes['a'], nodes['j']).name == 'l'
    assert find_successor(nodes['a'], nodes['l']).name == 'm'
    assert find_successor(nodes['a'], nodes['m']).name == 'k'
    assert find_successor(nodes['a'], nodes['k']).name == 'n'
    assert find_successor(nodes['a'], nodes['n']).name == 'i'
    assert find_successor(nodes['a'], nodes['i']).name == 'o'
    assert find_successor(nodes['a'], nodes['o']).name == 'p'


# pytest -s EPI\tests\test_binary_tree.py::test_find_successor_epi
def test_find_successor_epi():
    nodes = build_tree_dict_with_parent()
    assert find_successor_epi(nodes['d']).name == 'c'
    assert find_successor_epi(nodes['c']).name == 'e'
    assert find_successor_epi(nodes['e']).name == 'b'
    assert find_successor_epi(nodes['b']).name == 'f'
    assert find_successor_epi(nodes['f']).name == 'h'
    assert find_successor_epi(nodes['h']).name == 'g'
    assert find_successor_epi(nodes['g']).name == 'a'
    assert find_successor_epi(nodes['a']).name == 'j'
    assert find_successor_epi(nodes['j']).name == 'l'
    assert find_successor_epi(nodes['l']).name == 'm'
    assert find_successor_epi(nodes['m']).name == 'k'
    assert find_successor_epi(nodes['k']).name == 'n'
    assert find_successor_epi(nodes['n']).name == 'i'
    assert find_successor_epi(nodes['i']).name == 'o'
    assert find_successor_epi(nodes['o']).name == 'p'


# 9.13 Reconstruct a binary tree with markers
# pytest -s EPI\tests\test_binary_tree.py::test_build_pre_order_tree
def test_build_pre_order_tree():
    pre_order = ['H', 'B', 'F', None, None, 'E', 'A', None, None, None, 'C', None, 'D', None, 'G', 'I', None, None,
                 None]
    q = Queue(pre_order)
    node = build_pre_order_tree(q)
    assert node.data == 'H'
    assert node.left.data == 'B'
    assert node.right.data == 'C'


# 9.15 build leaves and edges
# pytest -s EPI\tests\test_binary_tree.py::test_build_leaves_and_edges
def test_build_leaves_and_edges():
    nodes = build_tree_dict_with_parent()
    res = build_leaves_and_edges(nodes['a'])
    assert len(res) == 11


# 9.16 Fill RLink
# pytest -s EPI\tests\test_binary_tree.py::test_build_rlink
def test_build_rlink():
    node = build_perfect_tree()
    build_rlink(node['a'])
    assert node['b'].rlink.data == 'i'


# 9.17 Build lock API
# pytest -s EPI\tests\test_binary_tree.py::test_lock_node
def test_lock_node():
    node = build_tree_dict_with_parent()
    assert lock_node(node['d']) == True
    assert lock_node(node['e']) == True
    assert lock_node(node['h']) == True
    assert lock_node(node['m']) == True
    assert lock_node(node['n']) == True
    assert lock_node(node['p']) == True
    assert lock_node(node['c']) == False
    assert lock_node(node['g']) == False
    assert lock_node(node['l']) == False
    assert lock_node(node['k']) == False

    assert unlock_node(node['m']) == True
    assert unlock_node(node['n']) == True
    assert lock_node(node['k']) == True
    assert lock_node(node['l']) == False

from ..graphs import *


# 18.0 Implement depth first search
# pytest -s EPI\tests\test_graph.py::test_dfs
def test_dfs():
    pass
    # res = optimize_task_assignment([10, 9, 8, 1])
    # assert res[0].task_1 == 1
    # assert res[0].task_2 == 10


# 18.4 Deadlock detection
# pytest -s EPI\tests\test_graph.py::test_find_a_deal_lock
def test_find_a_deal_lock():
    node_A = build_deadlock_graph()
    res = find_a_deal_lock(node_A)
    assert res == 1


# 18.6 Breadth first for bipartitian
# pytest -s EPI\tests\test_graph.py::test_can_be_bipartitian
def test_can_be_bipartitian():
    node_A = build_bi_partitian_graph()
    res = can_be_bipartitian(node_A)
    assert res == False


# 18.51 Shortest Path. Dijkstra's algorithm
# pytest -s EPI\tests\test_graph.py::test_shortest_path
def test_shortest_path():
    node_A = build_dijkstras_graph()
    res = shortest_path(node_A)
    assert res == 8


# 18.52 Test disjoint set
# pytest -s EPI\tests\test_graph.py::test_disjoint_set
def test_disjoint_set():
    ds = DisjointSet()
    ds.make_set('A')
    ds.make_set('B')

    ds.make_set('C')
    ds.make_set('D')

    ds.union('A', 'B')
    ds.union('C', 'D')
    res = ds.union('C', 'B')
    assert res.name == 'C'


# 18.52 Topological order
# pytest -s EPI\tests\test_graph.py::test_topological_order
def test_topological_order():
    res = topological_order(build_topo_graph())
    assert len(res) == 8


# 18.54 Floyd Warshal Algorithm. Shortest path from all nodes to all node
# pytest -s EPI\tests\test_graph.py::test_all_node_shortest_path
def test_all_node_shortest_path():
    g = build_floyd_warshal_graph()
    res = all_node_shortest_path(g)
    assert not res[1]
    assert res[0][0] == [0, 3, 1, 3]
    assert res[0][1] == [1, 0, -2, 0]
    assert res[0][2] == [3, 6, 0, 2]
    assert res[0][3] == [1, 4, 2, 0]


# 18.55 Kruskals algorith for MST , by pick all the edges and adding them to disjoint set
# pytest -s EPI\tests\test_graph.py::test_kruskals_min_spanning_tree
def test_kruskals_min_spanning_tree():
    nodes, edges = build_kruskals_graph()
    res = kruskals_min_spanning_tree(nodes, edges)
    assert res == 16


# 18.56 Prims algorith for MST , use a heap to get min and process the elements
# Logic ->  Push to heap and pick the element with min distance
# Traverse the neighbour and update their min value
# pytest -s EPI\tests\test_graph.py::test_prims_min_spanning_tree
def test_prims_min_spanning_tree():
    root, vertices = build_prims_graph()
    res = prims_min_spanning_tree(root, vertices)
    assert res == 9


# 18.57 Bellman ford , relaxes all not v-1 time, and in v-th time
# if it still decreases then there is negative weight cycle
# pytest -s EPI\tests\test_graph.py::test_bellman_ford_single_source_shortest_path
def test_bellman_ford_single_source_shortest_path():
    root, vertices, edges = build_bellman_ford_graph()
    res = bellman_ford_single_source_shortest_path(root, vertices, edges, 5)
    assert res == 5

# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment

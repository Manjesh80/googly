from enum import Enum
from heapq import *
import sys


class Vertex(object):
    def __init__(self, *, name=None, i=None, j=None, edges=None, data=None):
        self.parent = None
        self.name = name
        self.data = data
        self.rank = None
        self.visited = False
        self.weight = sys.maxsize
        self.min_distance = sys.maxsize
        self.neighbours = []
        self.edges = edges
        self.in_active_path = False
        self.i = i
        self.j = j
        self.color = None
        self.allowedColor = None
        self.visit_time = None
        self.low_time = None
        self.children = 0

    def __cmp__(self, other):
        return self.cmp(self.min_distance, other.min_distance)

    def __lt__(self, other):
        return self.min_distance < other.min_distance

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def add_neighbours(self, neighbours):
        self.neighbours.extend(neighbours)

    def get_matrix_name(self, r, c):
        return str((c * self.i) + self.j)

    def get_matrix_neighbours(self, r, c):
        res = []
        if self.i < r - 1:
            res.append(Vertex(i=self.i + 1, j=self.j))
        if self.j < c - 1:
            res.append(Vertex(i=self.i, j=self.j + 1))
        return res

    def get_neighbours(self):
        return self.neighbours


class Node(object):
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.color = 0
        self.visited = False
        self.blocked = False


class Edge(object):
    def __init__(self, weight, start, end):
        self.weight = weight
        self.start = start
        self.end = end


def dfs(root):
    def traverse(node):
        if not node.visited:
            node.visited = True
            print(f" Visited Node ==> {node.name}")
            for neighbour in node.get_neighbours():
                traverse(neighbour)

    traverse(root)


def traverse_maze(maze, r, c):
    def traverse(node):
        if not maze[node.i][node.j].blocked and destination_reached[0] == 0:
            print(f" Entering Node ==> {node.get_matrix_name(r,c)}")
            maze[node.i][node.j] = 1
            print(f" Visited Node ==> {node.get_matrix_name(r,c)}")
            if node.i == r - 1 and node.j == c - 1:
                print(f"Destination reached")
                destination_reached[0] = 1
                return
            for neighbour in node.get_matrix_neighbours(r, c):
                traverse(neighbour)
            print(f" Exiting Node ==> {node.get_matrix_name(r,c)}")

    destination_reached = [0]
    root = Vertex(name=str(0), i=0, j=0)
    traverse(root)


def build_maze(r, c):
    maze = [[Node(i, j) for j in range(c)] for i in range(r)]
    maze[2][1].blocked = True
    # maze[1][1].blocked = True
    # maze[0][1].blocked = True
    return maze


def build_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')

    node_A.add_neighbour(node_B)
    node_A.add_neighbour(node_C)
    node_A.add_neighbour(node_E)

    # node_B.add_neighbour(node_D)
    # node_C.add_neighbour(node_D)

    node_C.add_neighbour(node_E)

    # node_D.add_neighbour(node_F)
    # node_E.add_neighbour(node_F)
    return node_A


def build_bi_partitian_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')

    node_A.add_neighbour(node_B)
    node_A.add_neighbour(node_C)
    node_A.add_neighbour(node_E)
    # node_B.add_neighbour(node_D)
    # node_C.add_neighbour(node_D)
    node_C.add_neighbour(node_E)
    # node_D.add_neighbour(node_F)
    # node_E.add_neighbour(node_F)
    return node_A


def build_deadlock_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')

    edge_a_b = Edge(0, node_A, node_B)
    edge_a_c = Edge(0, node_A, node_C)
    node_A.edges = [edge_a_b, edge_a_c]

    edge_b_d = Edge(0, node_B, node_D)
    edge_b_e = Edge(0, node_B, node_E)
    node_B.edges = [edge_b_d, edge_b_e]

    edge_c_e = Edge(0, node_C, node_E)
    node_C.edges = [edge_c_e]

    edge_d_f = Edge(0, node_D, node_F)
    node_D.edges = [edge_d_f]

    edge_e_f = Edge(0, node_E, node_F)
    node_E.edges = [edge_e_f]

    edge_f_a = Edge(0, node_F, node_A)
    node_F.edges = [edge_f_a]

    node_A.add_neighbour(node_B)
    node_A.add_neighbour(node_C)

    node_B.add_neighbour(node_D)
    node_C.add_neighbour(node_E)

    node_D.add_neighbour(node_F)
    node_E.add_neighbour(node_F)
    return node_A


def find_a_deal_lock_with_path_print(root):
    def dfs(current):
        if current.name not in counter_set:
            counter_set.add(current.name)
        else:
            print(f"!!! Cycle detected => {current.name}")
            locked[0] = 1
            return
        if current.edges:
            for edge in current.edges:
                if locked[0] == 0:
                    dfs(edge.end)
        else:
            print(f" Path ==> {counter_set}")
        counter_set.remove(current.name)

    locked = [0]
    counter_set = set()
    dfs(root)


# 18.4 Deadlock detection
def find_a_deal_lock(root):
    def dfs(current):
        if not current.in_active_path:
            current.in_active_path = True
        else:
            print(f"!!! Cycle detected => {current.name}")
            locked[0] = 1
            return
        if current.edges:
            for edge in current.edges:
                if locked[0] == 0:
                    dfs(edge.end)
        current.in_active_path = False

    locked = [0]
    dfs(root)
    return locked[0]


class Color(Enum):
    White = 1
    Black = 2


# 18.6 Breadth first for bipartitian
# Breadth first with allowed color
def can_be_bipartitian(root):
    processing_queue = []
    root.allowedColor = Color.White
    processing_queue.append(root)
    is_bipartitian = True

    while processing_queue and is_bipartitian:
        root = processing_queue.pop(0)
        root.color = root.allowedColor
        if root.neighbours:
            neighbour_allowedColor = Color.White if root.color == Color.Black else Color.Black
            for neighbour in root.neighbours:
                if neighbour.allowedColor != None and neighbour.allowedColor != neighbour_allowedColor:
                    print("Bipartial is not possible")
                    is_bipartitian = False
                    break
                neighbour.allowedColor = neighbour_allowedColor
                processing_queue.append(neighbour)

    return is_bipartitian


def build_dijkstras_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')

    edge_a_b = Edge(1, node_A, node_B)
    edge_a_c = Edge(2, node_A, node_C)
    node_A.edges = [edge_a_b, edge_a_c]

    edge_b_d = Edge(5, node_B, node_D)
    edge_b_e = Edge(6, node_B, node_E)
    node_B.edges = [edge_b_d, edge_b_e]

    edge_c_e = Edge(3, node_C, node_E)
    node_C.edges = [edge_c_e]

    edge_d_f = Edge(2, node_D, node_F)
    node_D.edges = [edge_d_f]

    edge_e_f = Edge(10, node_E, node_F)
    node_E.edges = [edge_e_f]

    return node_A


def build_dijkstras_cyclic_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')

    edge_a_b = Edge(1, node_A, node_B)
    node_A.edges = [edge_a_b]

    edge_b_c = Edge(5, node_B, node_C)
    node_B.edges = [edge_b_c]

    edge_c_a = Edge(3, node_C, node_A)
    node_C.edges = [edge_c_a]

    return node_A


# 18.51 Shortest Path. Dijkstra's algorithm, do a breath-first greedy using heap
def shortest_path(root):
    processing_queue = []
    root.min_distance = 0
    processing_queue.append((root.min_distance, root, root.name))

    min_distance = 0

    while len(processing_queue) > 0:
        curr = heappop(processing_queue)
        if curr[1].edges:
            for edge in curr[1].edges:
                end_node = edge.end
                end_node_cur_min_distance = end_node.min_distance
                end_node_new_min_distance = curr[1].min_distance + edge.weight
                if end_node_new_min_distance < end_node_cur_min_distance:
                    end_node.min_distance = end_node_new_min_distance
                    heappush(processing_queue, (end_node_new_min_distance, edge.end, edge.end.name))
        else:
            print(f"Min distance is ==> {curr[1].min_distance}")
            min_distance = curr[1].min_distance
            break

    return min_distance


class DisjointSet(object):

    def __init__(self):
        self.store = {}

    def make_set(self, data):
        set_exists = self.find(data)
        if not set_exists:
            node = Vertex(name=data, data=data)
            node.parent = node
            node.rank = 0
            self.store[node.data] = node
            return node
        return set_exists

    def union(self, data1, data2):
        set1 = self.find(data1)
        set2 = self.find(data2)

        if not set1 and not set2:
            set1 = self.make_set(data1)
            set2 = self.make_set(data2)
        elif not set1:
            set1 = self.make_set(data1)
        elif not set2:
            set2 = self.make_set(data2)
        elif set1 is set2:
            return None

        if set1.rank >= set2.rank:
            self.merge_set(set1, set2)
            return set1
        else:
            self.merge_set(set2, set1)
            return set2

    def merge_set(self, parent, child):
        child_rep = self.find(child.data)
        child_rep.parent = parent
        child_rep.rank = 0
        parent.rank = 1

    def find(self, data):
        node = self.store.get(data, None)
        if node:
            node_parent = node.parent
            while node_parent.parent != node_parent:
                node_parent = node_parent.parent
            # Path compression
            if node.parent is not node_parent:
                node.parent = node_parent
                node.rank = 0
            return node_parent
        return node


# 18.52 Topological order
def topological_order(root):
    processing_queue = []
    processing_queue.append(root)
    topo = []

    while len(processing_queue) > 0:
        curr = processing_queue.pop(0)
        topo.append(curr.name)
        if curr.neighbours:
            for neighbour in curr.neighbours:
                processing_queue.append(neighbour)
    return topo


def build_topo_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')
    node_G = Vertex(name='G')
    node_O = Vertex(name='O')

    node_O.neighbours = [node_A, node_B]
    node_A.neighbours = [node_C]
    node_B.neighbours = [node_C, node_D]
    node_C.neighbours = [node_E]
    node_E.neighbours = [node_F]
    node_D.neighbours = [node_F]
    node_F.neighbours = [node_G]

    return node_B


def build_floyd_warshal_graph():
    node_0 = Vertex(name=0)
    node_1 = Vertex(name=1)
    node_2 = Vertex(name=2)
    node_3 = Vertex(name=3)

    edge_0_1 = Edge(3, node_0, node_1)
    edge_0_2 = Edge(6, node_0, node_2)
    edge_0_3 = Edge(15, node_0, node_3)
    node_0.edges = [edge_0_1, edge_0_2, edge_0_3]

    edge_1_2 = Edge(-2, node_1, node_2)
    node_1.edges = [edge_1_2]

    edge_2_3 = Edge(2, node_2, node_3)
    node_2.edges = [edge_2_3]

    edge_3_0 = Edge(1, node_3, node_0)
    node_3.edges = [edge_3_0]

    return [node_0, node_1, node_2, node_3]


# 18.54 Floyd Warshal Algorithm. Shortest path from all nodes to all node
# running for i ,j , k for n-cube time complexity
def all_node_shortest_path(vertices):
    def build_distance_matrix():
        for i, vertex in enumerate(vertices):
            distance_matrix[i][i] = 0
            for edge in vertex.edges:
                distance_matrix[i][edge.end.name] = edge.weight

    def build_route_matrix():
        for i, vertex in enumerate(vertices):
            for edge in vertex.edges:
                path_matrix[i][edge.end.name] = vertex.name

    nodes_count = len(vertices)
    distance_matrix = [[sys.maxsize for _ in range(nodes_count)] for _ in range(nodes_count)]
    path_matrix = [[-1 for _ in range(nodes_count)] for _ in range(nodes_count)]
    build_distance_matrix()
    build_route_matrix()

    for k in range(nodes_count):
        for i in range(nodes_count):
            for j in range(nodes_count):
                if distance_matrix[i][j] > (distance_matrix[i][k] + distance_matrix[k][j]):
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                    path_matrix[i][j] = path_matrix[k][j]

    cycle_detected = any([distance_matrix[v][v] < 0 for v in range(nodes_count)])
    return (distance_matrix, cycle_detected)


def build_kruskals_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')
    node_G = Vertex(name='G')

    edge_a_b = Edge(2, node_A, node_B)
    edge_a_c = Edge(6, node_A, node_C)
    edge_a_e = Edge(5, node_A, node_E)
    edge_a_f = Edge(10, node_A, node_F)

    node_A.edges = [edge_a_b, edge_a_c, edge_a_e, edge_a_f]

    edge_b_d = Edge(3, node_B, node_D)
    edge_b_e = Edge(3, node_B, node_E)

    node_B.edges = [edge_b_d, edge_b_e]

    edge_c_d = Edge(1, node_C, node_D)
    edge_c_f = Edge(2, node_C, node_F)
    node_C.edges = [edge_c_d, edge_c_f]

    edge_e_d = Edge(4, node_E, node_D)
    node_E.edges = [edge_e_d]

    edge_f_g = Edge(5, node_F, node_G)
    node_F.edges = [edge_f_g]

    edge_g_d = Edge(5, node_G, node_D)
    node_G.edges = [edge_g_d]

    edges = [edge_a_b, edge_a_c, edge_a_e, edge_a_f, edge_b_d, edge_b_e, edge_c_d, edge_c_f, edge_e_d, edge_f_g,
             edge_g_d]
    edges.sort(key=lambda e: e.weight)

    return ([node_A, node_B, node_C, node_D, node_E, node_F, node_G], edges)


# 18.55 Kruskals algorith for MST , by pick all the min edges and adding them to disjoint set
def kruskals_min_spanning_tree(nodes, edges):
    ds = DisjointSet()
    sum = 0
    for edge in edges:
        start_set = ds.make_set(edge.start)
        end_set = ds.make_set(edge.end)
        if ds.union(start_set, end_set):
            print(f"Edge added to {edge.start.name} ==> {edge.weight} ==> {edge.end.name}")
            sum += edge.weight
        else:
            print(f" !!! Edge Dropped to {edge.start.name} ==> {edge.weight} ==> {edge.end.name}")
    return sum


def build_prims_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')

    edge_a_b = Edge(3, node_A, node_B)
    edge_a_d = Edge(1, node_A, node_D)
    node_A.edges = [edge_a_b, edge_a_d]

    edge_b_a = Edge(3, node_B, node_A)
    edge_b_c = Edge(1, node_B, node_C)
    edge_b_d = Edge(3, node_B, node_D)
    node_B.edges = [edge_b_a, edge_b_c, edge_b_d]

    edge_c_b = Edge(1, node_C, node_B)
    edge_c_d = Edge(1, node_C, node_D)
    edge_c_e = Edge(5, node_C, node_E)
    edge_c_f = Edge(4, node_C, node_F)
    node_C.edges = [edge_c_b, edge_c_d, edge_c_e, edge_c_f]

    edge_d_a = Edge(1, node_D, node_A)
    edge_d_b = Edge(3, node_D, node_B)
    edge_d_c = Edge(1, node_D, node_C)
    edge_d_e = Edge(6, node_D, node_E)
    node_D.edges = [edge_d_a, edge_d_b, edge_d_c, edge_d_e]

    edge_e_d = Edge(6, node_E, node_D)
    edge_e_c = Edge(5, node_E, node_C)
    edge_e_f = Edge(2, node_E, node_F)
    node_E.edges = [edge_e_d, edge_e_c, edge_e_f]

    edge_f_c = Edge(4, node_F, node_C)
    edge_f_e = Edge(2, node_F, node_E)
    node_F.edges = [edge_f_c, edge_f_e]

    return (node_A, [node_A, node_B, node_C, node_D, node_E, node_F])


# 18.56 Prims algorith for MST , use a heap to get min and process the elements
# Logic ->  Push to heap and pick the element with min distance
# Traverse the neighbour and update their min value
# Grap from Tushar Roy video
# https://www.youtube.com/watch?v=oP2-8ysT3QQ&index=4&list=PLrmLmBdmIlpu2f2g8ltqaaCZiq6GJvl1j
def prims_min_spanning_tree(root, vertices):
    visited_set = set()
    processing_queue = []
    vertex_by_edge = {}
    root.min_distance = 0
    processing_queue.append((root.min_distance, root))
    heapify(processing_queue)

    while len(processing_queue) > 0:
        curr = heappop(processing_queue)
        visited_set.add(curr[1].name)

        for edge in curr[1].edges:
            if edge.end.name not in visited_set:
                new_min_distance = min(edge.end.min_distance, edge.weight)
                ele_in_list = [idx for idx, ele in enumerate(processing_queue) if ele[1].name == edge.end.name]
                if ele_in_list:
                    if new_min_distance < edge.end.min_distance:
                        processing_queue.pop(ele_in_list[0])
                        edge.end.min_distance = new_min_distance
                        vertex_by_edge[edge.end.name] = (edge.start.name, edge.end.name, edge.weight)
                        heappush(processing_queue, (edge.end.min_distance, edge.end))
                else:
                    vertex_by_edge[edge.end.name] = (edge.start.name, edge.end.name, edge.weight)
                    edge.end.min_distance = new_min_distance
                    heappush(processing_queue, (edge.end.min_distance, edge.end))

    [print(f" {k} ==> {v}") for k, v in vertex_by_edge.items()]
    sum = 0
    for k, v in vertex_by_edge.items():
        sum += v[2]
    return sum


def build_bellman_ford_graph():
    node_0 = Vertex(name=0)
    node_1 = Vertex(name=1)
    node_2 = Vertex(name=2)
    node_3 = Vertex(name=3)
    node_4 = Vertex(name=4)
    node_5 = Vertex(name=5)

    edge_0_1 = Edge(4, node_0, node_1)
    edge_0_2 = Edge(5, node_0, node_2)
    edge_0_3 = Edge(6, node_0, node_3)
    node_0.edges = [edge_0_1, edge_0_2, edge_0_3]

    edge_1_2 = Edge(-3, node_1, node_2)
    node_1.edges = [edge_1_2]

    edge_2_5 = Edge(4, node_2, node_5)
    node_2.edges = [edge_2_5]

    edge_3_4 = Edge(6, node_3, node_4)
    node_3.edges = [edge_3_4]

    edge_4_5 = Edge(2, node_4, node_5)
    node_4.edges = [edge_4_5]

    edge_5_4 = Edge(1, node_5, node_4)
    node_5.edges = [edge_5_4]

    return (node_0, [node_0, node_1, node_2, node_3, node_4, node_5],
            [edge_0_1, edge_0_2, edge_0_3, edge_1_2, edge_2_5, edge_3_4, edge_4_5, edge_5_4])


# 18.57 Bellman ford , relaxes all not v-1 time, and in v-th time
# if it still decreases then there is negative weight cycle
# https://www.youtube.com/watch?v=-mOEd_3gTK0&list=PLrmLmBdmIlpu2f2g8ltqaaCZiq6GJvl1j&index=6
def bellman_ford_single_source_shortest_path(source, vertices, edges, sink):
    source.min_distance = 0
    source_to_distance = {vertex.name: vertex.min_distance for vertex in vertices}
    for _ in range(1, len(vertices)):
        for edge in edges:
            if source_to_distance[edge.end.name] > source_to_distance[edge.start.name] + edge.weight:
                source_to_distance[edge.end.name] = source_to_distance[edge.start.name] + edge.weight
                edge.end.parent = edge.start

    cycle_found = any(
        [source_to_distance[edge.end.name] > source_to_distance[edge.start.name] + edge.weight for edge in edges])
    print(f" Cycle found ==> {cycle_found}")
    [print(f" {k} ==> {v}") for k, v in source_to_distance.items()]
    return source_to_distance[sink]


def build_tarjan_graph():
    node_A = Vertex(name='A')
    node_B = Vertex(name='B')
    node_C = Vertex(name='C')
    node_D = Vertex(name='D')
    node_E = Vertex(name='E')
    node_F = Vertex(name='F')

    node_A.add_neighbours([node_B, node_C])
    node_B.add_neighbours([node_E, node_A, node_C])
    node_C.add_neighbours([node_D, node_B, node_E, node_A])
    node_D.add_neighbours([node_C])
    node_E.add_neighbours([node_C, node_B])

    return node_A


# def build_tarjan_graph():
#     node_A = Vertex(name='A')
#     node_B = Vertex(name='B')
#     node_C = Vertex(name='C')
#
#     node_A.add_neighbour(node_B)
#     node_B.add_neighbour(node_A)
#
#     node_A.add_neighbour(node_C)
#     node_C.add_neighbour(node_A)
#
#     node_B.add_neighbour(node_C)
#     node_C.add_neighbour(node_B)
#     return node_A


# 18.58 Tarjan articulation point, to find single point of failure
# find the visited time and low-visit-time, and apply 2 rules
# if you are root and have 2 child or
# if have visit time lesser that low time of adjacent vertex
# When you find back path update the current vertex low time to min of adjacent and update parents low time as well
def tarjans_articulation_point(root):
    visited = set()
    articualtion_points = set()

    def tarjan(node: Vertex):
        visit_time[0] += 1
        node.visit_time = node.low_time = visit_time[0]
        visited.add(node.name)
        print(f"Processing Node ==> {node.name} Set visit time ==> {node.visit_time} ==> low time ==> {node.low_time}")
        for neighbour in node.neighbours:
            # Neighbour is parent ignore
            if node.parent and neighbour.name == node.parent.name:
                continue
            # its a back path perform logic else set visited time and low time and move on
            if neighbour.name in visited:
                # Get min of visit time of all , and update low-time
                node.low_time = min([neighbour.visit_time for neighbour in node.neighbours])
                if node.parent:
                    new_low_time = min(
                        [neighbour.visit_time for neighbour in node.parent.neighbours if
                         neighbour.visit_time is not None])
                    print(
                        f"Node ==> {node.name}  setting parent low_time ==> "
                        f"{node.parent.name} from low time ==> {node.parent.low_time} to ==> {new_low_time}")
                    node.parent.low_time = new_low_time
            else:
                node.children += 1
                neighbour.parent = node
                tarjan(neighbour)

        if node.parent is not None and node.visit_time <= max(
                [neighbour.low_time for neighbour in node.neighbours if neighbour.low_time is not None]):
            articualtion_points.add(node.name)

        if node.parent is None and node.children > 2:
            articualtion_points.add(node.name)

    visit_time = [0]
    tarjan(root)
    print(articualtion_points)
    return articualtion_points


if __name__ == "__main__":
    root = build_tarjan_graph()
    res = tarjans_articulation_point(root)
    print(res)

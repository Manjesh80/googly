from collections import defaultdict
from collections import Counter
from functools import reduce
from collections import OrderedDict
from heapq import *


class Employee_Test:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        # return True
        return self.name == other.name

    def __hash__(self):
        return 1


class ISBN:
    def __init__(self, isbn, price):
        self.isbn = isbn
        self.price = price


def test_bad_hash():
    e1 = Employee_Test("Ganesh")
    e2 = Employee_Test("Suresh")
    d = OrderedDict()
    d[e1] = e1
    d[e2] = e2
    # d = {e1: e1, e2: e2}
    print(d[e1].name)
    print(d[e2].name)


# words = ["levis", "elvis", "silent", "listen", "money"]
# find_anagrams(words)
def find_anagrams(words):
    sorted_words = defaultdict(list)
    for word in words:
        sorted_words[''.join(sorted(word))].append(word)
    [print(sorted_words[k] if len(sorted_words[k]) > 1 else '') for k in sorted_words.keys()]


def compute_string_hash(word, modulus):
    MULT = 9
    char_hash = lambda carry, character: (carry * MULT + ord(character)) % modulus
    res = reduce(char_hash, word, 0)
    return res


# 12.1 Test if a value can be permuted for palindromcity
def can_string_be_permuted_for_palindromicity(word):
    return True if sum([i % 2 for i in Counter(word).values()]) >= 1 else False


# 12.3 Implement ISBN LRU Search
class LRUCache:
    def __init__(self, capacity):
        self.store = OrderedDict()
        self.capacity = capacity

    def lookup(self, isbn):
        if isbn not in self.store:
            return False, None
        else:
            price = self.store.pop(isbn)
            self.store[isbn] = price
            return True, price

    def insert(self, isbn, price):
        if isbn in self.store:
            price = self.store.pop(isbn)
            self.store[isbn] = price
        elif len(self.store) == self.capacity:
            self.store.popitem(False)
        self.store[isbn] = price

    def erase(self, isbn):
        return self.store.pop(isbn, None) is not None

    def __str__(self):
        return str(self.store)


# 12.4 Compute the LCA with Parent( lowest common ancestor ) in Binary Tree
def lca_with_parent_optimized(node0, node1):
    nodes_on_way = set()

    # while node0 or node1:
    #     found = node0_parents.intersection(node1_parents)
    #     if len(found) == 1:
    #         parent = found.pop
    #         break
    #     else:
    #         node0_parents.add(node0.parent)
    #         node1_parents.add(node1.parent)
    #
    #     node0, node1 = node0.parent, node1.parent
    while node0 or node1:
        if node0:
            if node0 in nodes_on_way:
                return node0
            nodes_on_way.add(node0)
            node0 = node0.parent
        if node1:
            if node1 in nodes_on_way:
                return node1
            nodes_on_way.add(node1)
            node1 = node1.parent
    return ValueError("No Common parent")


# 12.5 Compute K most frequent queries
def k_most_frequent_queries(queries, k):
    c = Counter(queries)
    top_k = []
    for c_k_v in c.items():
        if len(top_k) < k:
            heappush(top_k, (c_k_v[1], c_k_v[0]))
        else:
            poppie = heappushpop(top_k, (c_k_v[1], c_k_v[0]))
    return top_k


if __name__ == "__main__":
    queries = ['Query1', 'Query2', 'Query3', 'Query4', 'Query1', 'Query2', 'Query2',
               'Query1', 'Query4', 'Query1', 'Query4', 'Query1', 'Query4', 'Query1',
               'Query1', 'Query4', 'Query1', 'Query4', 'Query1', 'Query4', 'Query1']
    res = k_most_frequent_queries(queries,2)
    print(res)
    # top_k = []
    # for c_k_v in c.items():
    #     if len(top_k) < 2:
    #         heappush(top_k, (c_k_v[1], c_k_v[0]))
    #     else:
    #         poppie = heappushpop(top_k, (c_k_v[1], c_k_v[0]))
    #         print(f"Evicted ==> {poppie[1]} -> {poppie[0]}")
    #
    # print("************ - *********** ")
    # print(top_k)
    # print("************ - *********** ")
    # print(c.most_common(2))
    # print("************ - *********** ")

    # top_queries = k_most_frequent_queries()

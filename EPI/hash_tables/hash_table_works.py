from collections import defaultdict
from collections import Counter
from functools import reduce
from collections import OrderedDict
from heapq import *
import sys


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


# 12.7 Find the smallest Sub Array covering all values
def find_smallest_subarray_covering_subset(stream, query_strings):
    class DoublyLinkedListNode:
        def __init__(self, data=None):
            self.data = data
            self.prev = self.next = None

    class LinkedList:
        def __init__(self):
            self.head = self.tail = None
            self._size = 0

        def __len__(self):
            return self._size

        def insert_after(self, value):
            node = DoublyLinkedListNode(value)
            node.prev = self.tail
            if self.tail:
                self.tail.next = node
            else:
                self.head = node
            self.tail = node
            self._size += 1

        def remove(self, node):
            if node.next:
                node.next.prev = node.prev
            else:
                self.tail = node.prev

            if node.prev:
                node.prev.next = node.next
            else:
                self.head = node.next
            node.next = node.prev = None
            self._size -= 1

        def __repr__(self):
            return self.str_rep()

        def __str__(self):
            return self.str_rep()

        def str_rep(self):
            try:
                res = ""
                curr = self.head
                while curr:
                    res += str(curr.data) + "( " + str(stream[curr.data]) + " ) " + " ==> "
                    curr = curr.next
                return res
            except:
                e = sys.exc_info()[0]
                print(e)

            return 'Got Error'

    loc = LinkedList()
    d = {s: None for s in query_strings}
    res = (-1, -1)

    for idx, s in enumerate(stream):
        if s in d:
            it = d[s]
            if it is not None:
                loc.remove(it)
            loc.insert_after(idx)
            d[s] = loc.tail

            if len(loc) == len(query_strings):
                if res == (-1, -1) or idx - loc.head.data < res[1] - res[0]:
                    res = (loc.head.data, idx)
    return res


# 12.7 Find the smallest Sub Array covering all values
def find_smallest_subarray_covering_subset_with_odict(stream, query_strings):
    loc = OrderedDict()
    d = {s for s in query_strings}
    res = (-1, -1)

    for idx, s in enumerate(stream):
        if s in d:
            loc.pop(s, None)
            loc[s] = idx
            if len(loc) == len(query_strings):
                start_index = next(iter(loc.items()))[1]
                if res == (-1, -1) or idx - start_index < res[1] - res[0]:
                    res = (start_index, idx)
    return res


# 12.8 Find the smallest Sub Array sequentially covering all values
def find_smallest_subarray_sequentially_covering_subset_with_odict(stream, query_strings):
    loc = OrderedDict()
    o_d = OrderedDict()
    for q_s in query_strings:
        o_d[q_s] = q_s
    res = (-1, -1)

    o_d_iter = iter(o_d.items())
    next_char = next(o_d_iter)[0]
    for idx, s in enumerate(stream):
        if s in o_d and s == next_char:
            loc.pop(s, None)
            loc[s] = idx
            if len(loc) == len(query_strings):
                start_index = next(iter(loc.items()))[1]
                if res == (-1, -1) or idx - start_index < res[1] - res[0]:
                    res = (start_index, idx)
                o_d_iter = iter(o_d.items())
                loc.clear()

            next_char = next(o_d_iter)[0]
    return res


# 12.9 Find longest sub-array with distinct entries
# ['f', 's', 'f', 'e', 't', 'w', 'e', 'n', 'w', 'e']
def find_the_longest_sub_array_with_distinct_entries(words):
    dist_dict = OrderedDict()
    result = (-1, -1)
    curr_result_start = 0
    for idx, w in enumerate(words):
        if w in dist_dict:
            curr_result = (curr_result_start, idx - 1)
            result = result if result_greater(result, curr_result) else curr_result
            curr_result_start = trim_left_until(w, dist_dict)
        else:
            dist_dict[w] = idx

    curr_result = (curr_result_start, len(words) - 1)
    result = result if result_greater(result, curr_result) else curr_result

    return result


def trim_left_until(w, dist_dict):
    new_start_index = 0
    while len(dist_dict) > 0:
        head = next(iter(dist_dict.items()))
        if w == head[0]:
            dist_dict.popitem(False)
            new_start_index = next(iter(dist_dict.items()))[1]
        else:
            dist_dict.popitem(False)
    return new_start_index


def result_greater(this, that):
    return this[1] - this[0] > that[1] - that[0]


# 12.10 Find the length of the longest contained interval
def len_of_longest_contained_interval(nums):
    nums = set(nums)
    long_run = 0
    while len(nums) > 0:
        current_value = nums.pop()
        lower_bound_count = 0
        lower_bound = current_value - 1
        while lower_bound in nums:
            nums.remove(lower_bound)
            lower_bound -= 1
            lower_bound_count += 1

        upper_bound = current_value + 1
        upper_bound_count = 0
        while upper_bound in nums:
            nums.remove(upper_bound)
            upper_bound += 1
            upper_bound_count += 1
        new_run = (lower_bound_count + upper_bound_count + 1)
        long_run = long_run if long_run > new_run else new_run

    return long_run


# 12.11 Find the student with top three scores
def top_three_scores(scores):
    student_dict = defaultdict(list)
    for score in scores:
        name, test_score = score.split(',')[0], int(score.split(',')[1])
        if len(student_dict[name]) >= 3:
            heappushpop(student_dict[name], test_score)
        else:
            student_dict[name].append(test_score)

    added_score = []
    for student_con in student_dict.items():
        if len(student_con[1]) <= 3:
            added_score.append(sum(student_con[1]))

    return max(added_score)


# 12.2 Compute all String decomposition
def compute_all_decomposition(sentence, words):
    unit_size = len(words[0])
    words_with_freq = Counter(words)

    def decomposition_present(start, end):
        sentence_word_frequency = {}
        for i in range(start, end, unit_size):
            curr_word = sentence[i: i + unit_size]
            if curr_word in words_with_freq:
                sentence_word_frequency[curr_word] = 1 if sentence_word_frequency.get(curr_word) is None else (
                        sentence_word_frequency.get(curr_word) + 1)
                curr_word_master_freq = words_with_freq[curr_word]
                curr_word_sentence_freq = sentence_word_frequency[curr_word]
                if curr_word_sentence_freq > curr_word_master_freq:
                    return False
                if len(sentence_word_frequency) == len(words_with_freq):
                    return True
            else:
                return False

    for i in range(0, len(sentence)):
        if decomposition_present(i, len(sentence)):
            return sentence[i: i + (len(words) * len(words[0]))]


if __name__ == "__main__":
    sentence = 'amanaplanacanapl'
    sentence = 'xxxcanaplanaaplzzz'
    words = ['can', 'apl', 'ana', 'apl']
    # sentence = 'can'
    # words = ['can']
    res = compute_all_decomposition(sentence, words)
    print(res)

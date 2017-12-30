class Node():
    def __init__(self, value, next_node, jump_node):
        self.value = value
        self.order = -1
        self.next_node = next_node
        self.jump_node = jump_node


# 8.5 Search a posting list
def search_posting_list(L):
    def search_post(N):
        if N and N.order == -1:
            order[0] += 1
            N.order = order[0]
            search_post(N.jump_node)
            search_post(N.next_node)

    order = [0]
    search_post(L)
    return


def search_posting_list_stack(L):
    s, order = [L], 0
    while s:
        curr = s.pop()
        if curr and curr.order == -1:
            curr.order = order
            order += 1
            s += [curr.next_node, curr.jump_node]


if __name__ == "__main__":
    d_node = Node('d', None, None)
    c_node = Node('c', None, None)
    b_node = Node('b', None, None)
    a_node = Node('a', None, None)

    a_node.jump_node = c_node
    a_node.next_node = b_node

    b_node.jump_node = d_node
    b_node.next_node = c_node

    c_node.jump_node = b_node
    c_node.next_node = d_node

    d_node.jump_node = d_node

    search_posting_list_stack(a_node)

    print(a_node)

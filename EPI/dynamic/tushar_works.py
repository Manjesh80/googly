# 40.1 Knapsack - Implement micro

# 40.2 Longest Common Sub sequence
def test_longest_common_subsequence():
    pass


# 40.4 Optimal Binary Search tree
# res = optimal_binary_search_tree([4, 2, 6, 3], None)
def optimal_binary_search_tree(nodes, rates):
    n = len(nodes)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i + j) < n:
                if j == (i + j):  # diagonal
                    result[j][i + j] = nodes[i + j]
                else:
                    current_sum = sum(nodes[j:i + j + 1])
                    k_roots = []
                    for k in range(j, i + j + 1):
                        print(f" If {k} is root for range( {j} --> {i + j} )")
                        if k < (i + j):
                            root_sum = result[j][k - 1] + result[k + 1][i + j]
                        else:
                            root_sum = result[j][k - 1]
                        k_roots.append(root_sum)
                    current_sum += min(k_roots)
                    result[j][i + j] = current_sum
    return result


def optimal_binary_search_tree_new(nodes, rates):
    n = len(nodes)
    result = [[-1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i + j) < n:
                result[j][i + j] = 100
                print(f'({j} , {i+j} )')
    return result


# 40.10 Minimum number of coins
def minimum_number_of_coins(coins, total):
    result = [float('inf') for _ in range(total + 1)]
    for i in range(total + 1):
        for k in range(len(coins)):
            if i == coins[k]:
                result[i] = min(result[i], 1)
            elif i > coins[k]:
                result[i] = min(result[i], result[i - coins[k]] + 1)

    print(result)


if __name__ == "__main__":
    res = minimum_number_of_coins([7, 3, 6, 13], 13)

import time


# https://leetcode.com/explore/interview/card/google/64/dynamic-programming-4/356/
# ws = WSolution()
# print(ws.wordsTyping(rows=2, cols=8, sentence=["hello", "world"]))
# print(ws.wordsTyping(rows=3, cols=6, sentence=["a", "bcd", "e"]))
# print(ws.wordsTyping(rows=4, cols=5, sentence=["I", "had", "apple", "pie"]))
class WSolution:
    def wordsTyping(self, sentence, rows, cols):
        word_len = {}
        num_of_words = len(sentence)
        for idx, word in enumerate(sentence):
            current_line_len = 0
            words_count = 0
            curr_word = idx
            while True:
                new_line_len = current_line_len + len(sentence[curr_word])
                if new_line_len > cols:
                    break
                current_line_len = new_line_len + 1  # Added space
                words_count += 1
                curr_word = 0 if (curr_word == (num_of_words - 1)) else (curr_word + 1)

            word_len[idx] = (words_count, curr_word)

        start_with, total_words = 0, 0
        for _ in range(rows):
            line_words, start_with = word_len[start_with]
            total_words += line_words
        return total_words // num_of_words


# https://leetcode.com/explore/interview/card/google/64/dynamic-programming-4/348/
# wb = WBSolution()
# print(wb.wordBreak(s="leetcode", wordDict=["leet", "code"]))
# print(wb.wordBreak(s="ab", wordDict=["a", "b"]))
# print(wb.wordBreak("abcd", ["a", "abc", "b", "cd"]))
class WBSolution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if not s or not wordDict:
            return False
        result = [False for _ in range(len(s))]
        idx = 0
        while idx < len(s):
            for word in wordDict:
                end_index = idx + len(word)
                if (word == s[idx:end_index]) and (True if (idx == 0) else result[idx - 1]):
                    result[end_index - 1] = True
            idx += 1
            if any(result[idx:]):
                while idx < len(s) and not result[idx]:
                    idx += 1
                idx += 1
        return result[-1]

    def wordBreak2(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if not s or not wordDict:
            return False
        result = [False for _ in range(len(s))]
        for idx in range(len(s)):
            for word in wordDict:
                end_index = idx + len(word)
                if (word == s[idx:end_index]) and (True if (idx == 0) else result[idx - 1]):
                    result[end_index - 1] = True
        return result[-1]


# https://leetcode.com/explore/interview/card/google/64/dynamic-programming-4/367/
# obj = MVSolution()
# print(obj.maxVacationDays(flights=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], days=[[1, 3, 1], [6, 0, 3], [3, 3, 3]]))
# print(obj.maxVacationDays(flights=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], days=[[1, 1, 1], [7, 7, 7], [7, 7, 7]]))
# print(obj.maxVacationDays(flights=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], days=[[7, 0, 0], [0, 7, 0], [0, 0, 7]]))
class MVSolution:
    def maxVacationDays(self, flights, days):
        """
        :type flights: List[List[int]]
        :type days: List[List[int]]
        :rtype: int
        """
        if len(flights) != len(days):
            return 0
        rows = len(days)
        cols = len(days[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for col in range((cols - 1), -1, -1):
            for row in range(rows):
                if col == (cols - 1):
                    result[row][col] = days[row][col]
                else:
                    curr_acc = result[row][col + 1]
                    max_res = curr_acc
                    for k in range(rows):
                        if (row == k) or (flights[row][k]):
                            max_res = max(max_res, days[row][col] + result[k][col + 1])
                    result[row][col] = max_res

        for row in range(rows):
            max_res = result[row][0]
            for k in range(rows):
                if (row == k) or (flights[row][k]):
                    max_res = max(max_res, result[k][0])
            result[row][col] = max_res
        return result[0][0]

    def maxVacationDays2(self, flights, days):
        """
        :type flights: List[List[int]]
        :type days: List[List[int]]
        :rtype: int
        """
        NINF = float('-inf')
        ROWS, COLS = len(days), len(days[0])
        best = [NINF] * ROWS
        best[0] = 0

        for t in range(COLS):
            cur = [NINF] * ROWS
            for i in range(ROWS):
                for j, adj in enumerate(flights[i]):
                    if adj or i == j:
                        cur[j] = max(cur[j], best[i] + days[j][t])
            best = cur
        return max(best)


one = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '*': 9}
two = {'10': 1, '11': 1, '12': 1, '13': 1, '14': 1, '15': 1, '16': 1, '17': 1, '18': 1, '19': 1, '20': 1, '21': 1,
       '22': 1, '23': 1, '24': 1, '25': 1, '26': 1, '*0': 2, '*1': 2, '*2': 2, '*3': 2, '*4': 2, '*5': 2, '*6': 2,
       '*7': 1, '*8': 1, '*9': 1, '1*': 9, '2*': 6, '**': 15}


# https://leetcode.com/explore/interview/card/google/64/dynamic-programming-4/334/
# Decode Ways
class NDSolution:
    def numDecodings_leet(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = 1, one.get(s[:1], 0)

        for i in range(1, len(s)):
            dp = dp[1], (one.get(s[i], 0) * dp[1] + two.get(s[i - 1: i + 1], 0) * dp[0]) % 1000000007

        return dp[-1]

    def numDecodings(self, s):
        if not s:
            return 0
        M = 1000000007;
        one_to_nine = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        one_to_six = ['1', '2', '3', '4', '5', '6']
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        bad_data = False
        for i in range(0, len(s)):
            prev_with_current_value = 0
            current_digit = s[i]
            prev_digit = s[i - 1] if (i - 1) > -1 else 'X'

            current_value = 9 if current_digit == '*' else 1
            if prev_digit != 'X':
                if prev_digit == '1':
                    if current_digit == '*':
                        prev_with_current_value = 9
                    elif current_digit in one_to_nine:
                        prev_with_current_value = 1
                elif prev_digit == '2':
                    if current_digit == '*':
                        prev_with_current_value = 6
                    elif current_digit in one_to_six:
                        prev_with_current_value = 1
                elif prev_digit == '*':
                    if current_digit == '*':
                        prev_with_current_value = 15
                    elif int(current_digit) < 7:
                        prev_with_current_value = 2
                    else:
                        prev_with_current_value = 1
                elif (prev_digit == '0' and current_digit == '0') or \
                        (int(prev_digit) > 3 and current_digit == '0'):
                    bad_data = True
                    break
            dp_idx = i + 1
            if current_digit == '0' and prev_digit == '*':
                dp[dp_idx] = (prev_with_current_value * dp[dp_idx - 2]) % M
            else:
                dp[dp_idx] = (current_value * dp[dp_idx - 1] + prev_with_current_value * dp[dp_idx - 2]) % M

        return dp[-1] if not bad_data else 0

    def numDecodings_part_1(self, s):
        if not s:
            return 0

        def decode(ss, i):
            if double_zero[0]:
                return False, 0
            if not ss:
                return True, 1

            left_result, right_result = None, None
            if len(ss) > 0:
                if int(ss[0]) > 0:
                    if str(i + 1) in memoize:
                        left_result = memoize[str(i + 1)]
                    else:
                        left_result = decode(ss[1:], i + 1)
                        memoize[str(i + 1)] = left_result
                else:
                    left_result = (True, 0)
            if len(ss) > 1:
                if (9 < int(ss[0:2]) < 27):
                    key = str(i + 1) + ',' + str(i + 2)
                    if key in memoize:
                        right_result = memoize[key]
                    else:
                        right_result = decode(ss[2:], i + 2)
                        memoize[key] = right_result
                elif int(ss[0:2]) == 0:
                    double_zero[0] = True
                    right_result = (False, 0)
                else:
                    right_result = (False, 0)
            else:
                right_result = (True, 0)
            if left_result and right_result:
                return ((left_result[0] or right_result[0]), (left_result[1] + right_result[1]))
            elif left_result or right_result:
                return left_result if left_result else right_result
            else:
                return (True, 0)

        memoize = {}
        double_zero = [False]
        valid, codes = decode(s, 0)
        result = codes if valid else 0
        return result

    def numDecodings_1(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or s.find('00') > 0:
            return 0
        L = len(s)
        result = [[0 for _ in range(L)] for _ in range(L)]
        invalid_nubmer = False
        for j in range(L):
            for i in range(L - j):
                r, c = i, i + j
                if r == c:
                    result[r][c] = 1 if int(s[r]) > 0 else 0
                else:
                    result[r][c] = min(result[r][c - 1], result[r + 1][c])
                    if int(s[r]) == 0:
                        result[r][c] += 0
                    if int(s[c]) == 0:
                        if int(s[r]) in (1, 2):
                            result[r][c] += 0
                        else:
                            invalid_nubmer = True
                            break
                    if (9 < int(s[r] + s[c]) < 27):
                        result[r][c] += 1
        if invalid_nubmer:
            return 0
        return result[0][L - 1]


class ND2Solution:
    def numDecodings_leet(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = 1, one.get(s[:1], 0)

        for i in range(1, len(s)):
            dp = dp[1], (one.get(s[i], 0) * dp[1] + two.get(s[i - 1: i + 1], 0) * dp[0]) % 1000000007

        return dp[-1]

    def numDecodings(self, s):
        if not s:
            return 0
        M = 1000000007;
        one_to_nine = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        one_to_six = ['1', '2', '3', '4', '5', '6']
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        bad_data = False
        for i in range(0, len(s)):
            prev_with_current_value = 0
            current_digit = s[i]
            prev_digit = s[i - 1] if (i - 1) > -1 else 'X'

            current_value = 9 if current_digit == '*' else 1
            if prev_digit != 'X':
                if prev_digit == '1':
                    if current_digit == '*':
                        prev_with_current_value = 9
                    elif current_digit in one_to_nine:
                        prev_with_current_value = 1
                elif prev_digit == '2':
                    if current_digit == '*':
                        prev_with_current_value = 6
                    elif current_digit in one_to_six:
                        prev_with_current_value = 1
                elif prev_digit == '*':
                    if current_digit == '*':
                        prev_with_current_value = 15
                    elif int(current_digit) < 7:
                        prev_with_current_value = 2
                    else:
                        prev_with_current_value = 1
                elif (prev_digit == '0' and current_digit == '0') or \
                        (int(prev_digit) > 3 and current_digit == '0'):
                    bad_data = True
                    break
            dp_idx = i + 1
            if current_digit == '0':
                if prev_digit == '*':
                    dp[dp_idx] = (prev_with_current_value * dp[dp_idx - 2]) % M
                elif prev_digit in {'1', '2'}:
                    dp[dp_idx] = (dp[dp_idx - 2]) % M
            else:
                dp[dp_idx] = (current_value * dp[dp_idx - 1] + prev_with_current_value * dp[dp_idx - 2]) % M

        return dp[-1] if not bad_data else 0


# https://leetcode.com/problems/regular-expression-matching/description/
class ExpMatchSolution:
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not s or not p:
            return False
        res = [[False for _ in range((len(p) + 1))] for _ in range(len(s) + 1)]
        for pi in range(len(p) + 1):
            # res[0][pi] = True if ((p[pi - 1] == '*') or (pi == 0)) else False
            res[0][pi] = True if pi == 0 else False
            # res[0][pi] = True if ( (p[pi - 1] == '*' and (pi != len(p))) or (pi == 0)) else False
        for si in range(len(s) + 1):
            res[si][0] = True if si == 0 else False

        for si in range(0, len(s)):
            for pi in range(0, len(p)):
                sidx = si + 1
                pidx = pi + 1
                if p[pi] == '.' or p[pi] == s[si]:
                    res[sidx][pidx] = res[sidx - 1][pidx - 1]
                elif p[pi] == '*':
                    res[sidx][pidx] = res[sidx][pidx - 2] or \
                                      (res[sidx][pidx - 1] if p[pi - 1] in {s[si], '.'} else False)
                else:
                    res[sidx][pidx] = False
        return res[-1][-1]

    def isMatch_old_1(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not s or not p:
            return False
        res = [[False for _ in range((len(p) + 1))] for _ in range(len(s) + 1)]
        for pi in range(len(p) + 1):
            res[0][pi] = True if ((p[pi - 1] == '*') or (pi == 0)) else False
            # res[0][pi] = True if ((p[pi - 1] == '*' and (pi != len(p))) or (pi == 0)) else False
        for si in range(len(s) + 1):
            res[si][0] = True if si == 0 else False

        for si in range(0, len(s)):
            for pi in range(0, len(p)):
                if p[pi] == '.' or p[pi] == s[si]:
                    res[si + 1][pi + 1] = res[si][pi]
                # elif p[pi] == s[si]:
                # if ((pi - 1) > -1) and (p[pi - 1] == '*'):
                #     res[si + 1][pi + 1] = False
                # else:
                # res[si + 1][pi + 1] = res[si][pi]
                elif p[pi] == '*':
                    res[si + 1][pi + 1] = res[si + 1][pi] or res[si][pi + 1]
                else:
                    res[si + 1][pi + 1] = False
        return res[-1][-1]

    def isMatch_old(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        si, pi = 0, 0
        ls, lp = len(s), len(p)
        result = True
        all_match_pattern = False
        while si < ls and pi < lp:
            cp, np = (p[pi], p[pi + 1]) if (pi + 1) < (lp) else (p[pi], None)
            if np == '*' and cp == '*':
                raise ValueError('Bad pattern')
                break
            if cp is '.' and np != '*':
                pi, si = pi + 1, si + 1
                continue
            if np == '*':
                if cp != '.':
                    while si < ls and s[si] == cp:
                        si += 1
                    pi += 2
                    while pi < lp and p[pi] == cp:
                        pi += 1
                    continue
                else:
                    si += 2
                    pi += 2
                    continue
            if cp == s[si]:
                pi, si = pi + 1, si + 1
            else:
                result = False
                break

        if all_match_pattern:
            return True
        if (si != ls) or (not result) or (pi != lp):
            return False
        else:
            return True


# https://leetcode.com/problems/wildcard-matching/description/
class WCSolution:
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not s and not p:
            return True
        if not s and p and len(p) == 1 and p[0] == '*':
            return True
        if not s or not p:
            return False
        res = [[False for _ in range((len(p) + 1))] for _ in range(len(s) + 1)]
        res[0][0] = True
        i = 0
        while i < len(p) and p[i] == '*':
            i += 1
            res[0][i] = True
        for pi in range(i + 1, len(p) + 1):
            res[0][pi] = False
        for si in range(1, len(s) + 1):
            res[si][0] = False

        for si in range(0, len(s)):
            for pi in range(0, len(p)):
                sidx = si + 1
                pidx = pi + 1
                if p[pi] == '?' or p[pi] == s[si]:
                    res[sidx][pidx] = res[sidx - 1][pidx - 1]
                elif p[pi] == '*':
                    res[sidx][pidx] = res[sidx][pidx - 1] or res[sidx - 1][pidx]
                else:
                    res[sidx][pidx] = False
        return res[-1][-1]


# https://leetcode.com/problems/maximal-rectangle/description/
class MRSolution:
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        rows = len(matrix)
        if not rows:
            return 0
        cols = len(matrix[0])
        if not cols:
            return 0

        def largestRectangleArea(height):
            increasing, area, i = [], 0, 0
            while i <= len(height):
                if not increasing or (i < len(height) and height[i] > height[increasing[-1]]):
                    increasing.append(i)
                    i += 1
                else:
                    last = increasing.pop()
                    if not increasing:
                        area = max(area, height[last] * i)
                    else:
                        area = max(area, height[last] * (i - increasing[-1] - 1))
            return area

        one = 1
        result = [0] * cols
        max_res = 0
        for r in range(rows):
            for c in range(cols):
                result[c] = (result[c] + 1) if (matrix[r][c]) == one else 0
            mauh = largestRectangleArea(result)
            max_res = max(max_res, mauh)
        return max_res

    def maximalRectangleOld(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        rows = len(matrix)
        if not rows:
            return 0
        cols = len(matrix[0])
        if not cols:
            return 0
        result = [[0] * (cols + 1) for _ in range(rows + 1)]
        max_res = 0
        one = 1
        for r in range(rows):
            for c in range(cols):
                r_r, r_c = r + 1, c + 1
                if matrix[r][c] == one:
                    top = matrix[r - 1][c] if r > 0 else 0
                    left = matrix[r][c - 1] if c > 0 else 0
                    diagonal = matrix[r - 1][c - 1] if c > 0 and r > 0 else 0
                    if top == left == diagonal == one:
                        if result[r_r - 1][r_c] == result[r_r][r_c - 1] == result[r_r - 1][r_c - 1] == 1:
                            new_res = result[r_r - 1][r_c] + result[r_r][r_c - 1] + result[r_r - 1][r_c - 1] + 1
                        else:
                            new_res = result[r_r - 1][r_c] + result[r_r][r_c - 1] + result[r_r - 1][r_c - 1]
                        result[r_r][r_c] = new_res
                        max_res = max(new_res, new_res)
                    else:
                        result[r_r][r_c] = 2 if matrix[r - 1][c] == '1' else 1
                else:
                    result[r_r][r_c] == 0
        print(result)
        return max_res


# https://leetcode.com/problems/distinct-subsequences/description/
# Distinct sub-sequence
class DSSolution:
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        if (not s and not t) or (s and not t):
            return 1
        if not s and t:
            return 0

        all_ones = [[1 for _ in range(len(s) + 1)]]
        result = all_ones + [[0] * (len(s) + 1) for _ in range(len(t))]

        for r in range(len(t)):
            for c in range(len(s)):
                r_r = r + 1
                r_c = c + 1
                if t[r] == s[c]:
                    result[r_r][r_c] = result[r_r - 1][r_c - 1] + result[r_r][r_c - 1]
                else:
                    result[r_r][r_c] = result[r_r][r_c - 1]
        return result[-1][-1]

    def numDistinctOld(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        if (not s and not t) or (s and not t):
            return 1
        if not s and t:
            return 0

        result = [[-1] * len(t) for _ in range(len(s))]

        def distinct(si, ti):
            if (si == -1 and ti == -1) or (si > -1 and ti == -1):
                return 1
            if si == -1 and ti > -1:
                return 0
            if result[si][ti] != -1:
                return result[si][ti]
            else:
                if s[si] == t[ti]:
                    result[si][ti] = distinct(si - 1, ti) + distinct(si - 1, ti - 1)
                else:
                    result[si][ti] = distinct(si - 1, ti)
                return result[si][ti]

        distinct(len(s) - 1, len(t) - 1)
        print(result)
        return result[-1][-1]


# https://leetcode.com/problems/longest-valid-parentheses/description/
class LVPSolution:
    def longestValidParentheses(self, s):
        s = ')' + s
        dp = [0] * len(s)
        temp = []
        result = 0
        for i in range(1, len(s)):
            if s[i] == ')' and not temp:
                dp[i] = 0
            elif s[i] == ')' and temp:
                last_open = temp.pop()
                dp[i] = (i - last_open + 1) + dp[last_open - 1]
                result = max(result, dp[i], dp[last_open - 1])
            elif s[i] == '(':
                temp.append(i)
        print(dp)
        return result

    def longestValidParentheses_fit(self, s):
        s = ')' + s
        dp = [0] * len(s)
        for i in range(1, len(s)):
            if s[i] == ')' and s[i - dp[i - 1] - 1] == '(':
                dp[i] = dp[i - 1] + 2 + dp[i - 2 - dp[i - 1]]
        return max(dp)

    def longestValidParentheses_3(self, s):
        if not s or len(s) == 1:
            return 0
        open_count = 0
        close_count = 0
        start_index, end_index, max_len = 0, 0, 0
        current_good_count = 0, 0
        for i in range(len(s)):
            is_open = True if s[i] == '(' else False
            if is_open:
                open_count += 1
            else:
                close_count += 1
            if open_count == close_count:
                current_good_count = max(current_good_count, open_count * 2)
            elif close_count > open_count:
                max_len = max(current_good_count, (min(open_count, close_count) * 2) if open_count > close_count else 0)
                open_count, close_count, current_good_count = 0, 0, 0

    def longestValidParentheses_new(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or len(s) == 1:
            return 0
        open_count = 0
        close_count = 0
        start_index, end_index, max_len = 0, 0, 0
        current_good_count = 0

        for i in range(len(s)):
            is_open = True if s[i] == '(' else False
            if is_open:
                open_count += 1
            else:
                close_count += 1
            if open_count == close_count:
                current_good_count = open_count * 2

            if close_count > open_count or i == (len(s) - 1):
                max_len = max(max_len, current_good_count,
                              (min(open_count, close_count) * 2) if open_count > close_count else 0)
                open_count, close_count, current_good_count = 0, 0, 0

        return max_len


# https://leetcode.com/problems/interleaving-string/description/
class ILSolution:
    def isInterleave_working(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1) + len(s2) != len(s3):
            return False
        if not s1 and s2 and s3:
            return True if s2 == s3 else False
        if not s2 and s1 and s3:
            return True if s1 == s3 else False
        if not s1 and not s2 and not s3:
            return True
        if (s1 and s2 and not s3) or (not s1 and not s2 and s3):
            return False

        result = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        result[0][0] = True
        for i in range(len(s1) + 1):
            il_idx = i - 1
            for j in range(len(s2) + 1):
                if i == j == 0:
                    continue
                if j > len(s3) - 1:
                    break
                s1_idx, s2_idx = i - 1, j - 1
                s1_char = s1[s1_idx] if s1_idx > -1 else None
                s2_char = s2[s2_idx] if s2_idx > -1 else None
                il_char = s3[il_idx + j]
                top = False if i == 0 else result[i - 1][j]
                left = False if j == 0 else result[i][j - 1]
                ds = set()
                if not left:
                    ds.add(s1_char)
                if not top:
                    ds.add(s2_char)
                result[i][j] = il_char in ds and (top or left)
                if (s1_char == s2_char == il_char) and top and left:
                    result[i][j] = True
                # print(f" [{i},{j}] ==> checking if ** {il_char} **  in  {ds} ==> set result to {result[i][j]}")
        # print(result)
        return result[-1][-1]

    def isInterleave(self, s1, s2, s3):
        if len(s1 + s2) != len(s3):
            return False

        def is_interleaved_rec(ss1, ss2, sil):
            if ss1 == -1:
                return True if s2[:ss2 + 1] == s3[:sil + 1] else False
            if ss2 == -1:
                return True if s1[:ss1 + 1] == s3[:sil + 1] else False
            if final_res[ss1][ss2] == -1:
                result = is_interleaved_rec(ss1 - 1, ss2, sil - 1) if s1[ss1] == s3[sil] else False
                result = result or (is_interleaved_rec(ss1, ss2 - 1, sil - 1) if s2[ss2] == s3[sil] else False)
                final_res[ss1][ss2] = result
            return final_res[ss1][ss2]

        final_res = [[-1] * len(s2) for _ in range(len(s1))]
        return is_interleaved_rec(len(s1) - 1, len(s2) - 1, len(s3) - 1)

    def isInterleave_old(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if not s1 and s2 and s3:
            return True if s2 == s3 else False
        if not s2 and s1 and s3:
            return True if s1 == s3 else False
        if not s1 and not s2 and not s3:
            return True
        if (s1 and s2 and not s3) or (not s1 and not s2 and s3):
            return False

        s1_i, s2_i, s3_i = 0, 0, 0
        while s1_i < len(s1) and s2_i < len(s2) and s3_i < len(s3):
            curr_s1 = s1[s1_i]
            curr_s2 = s2[s2_i]
            curr_s3 = s3[s3_i]

            if curr_s1 == curr_s3:
                s1_i, s3_i = s1_i + 1, s3_i + 1
            elif curr_s2 == curr_s3:
                s2_i, s3_i = s2_i + 1, s3_i + 1
            else:
                break

        r_s1, r_s2, r_s3 = str(s1[s1_i:]), str(s2[s2_i:]), str(s3[s3_i:])
        return True if sorted((r_s1 + r_s2)) == sorted(r_s3) else False


# https://leetcode.com/problems/palindrome-partitioning-ii/description/
class MCSolution:
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        ls = len(s)
        result = [[False] * ls for _ in range(ls)]
        for j in range(ls):
            for i in range(ls):
                r, c = i, i + j
                if r == c:
                    result[r][c] = True
                    continue
                if c < ls:
                    # this_string = s[r: c + 1]
                    result[r][c] = True if s[r: c + 1] == (s[r: c + 1])[::-1] else False
                else:
                    break

        final_result = [float('inf')] * ls
        for i in range(ls):
            if not result[0][i]:
                min_i_result = float('inf')
                for j in range(i + 1):
                    if result[j][i] == True:
                        min_i_result = min(min_i_result, final_result[j - 1] + 1)
                final_result[i] = min_i_result
            else:
                final_result[i] = 0

        return final_result[ls - 1]

    def minCut4(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        ls = len(s)
        result = [[-1] * ls for _ in range(ls)]
        for j in range(ls):
            for i in range(ls):
                r, c = i, i + j
                if r == c:
                    result[r][c] = 0
                    continue
                if c < ls:
                    # print(f"Processing [{r}][{c}]")
                    # this_string = str(s[r: c + 1])
                    # if this_string == this_string[::-1]:
                    if s[r: c + 1] == s[c:r - 1:-1]:
                        result[r][c] = 0
                    else:
                        min_k_count = float('inf')
                        for k in range(r, c):
                            k_res = result[r][k] + 1 + result[k + 1][c]
                            min_k_count = min(min_k_count, k_res)
                        result[r][c] = min_k_count
                else:
                    break

        # print(result)
        return result[0][ls - 1]

    def minCut_timoeout2(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        ls = len(s)
        result = [[-1] * len(s) for _ in range(len(s))]
        total_hit_count = [0]

        def palindromic_split(start, end):
            if end == ls:
                return
            if result[start][end] == -1:
                this_string = str(s[start:end + 1])
                if this_string == this_string[::-1]:
                    result[start][end] = 0
                else:
                    min_k_count = float('inf')
                    for k in range(start, end):
                        k_res = palindromic_split(start, k) + 1 + palindromic_split(k + 1, end)
                        min_k_count = min(min_k_count, k_res)
                    result[start][end] = min_k_count
            else:
                print(f"Hit found for Result[{start}][{end}]")
                total_hit_count[0] += 1
                return result[start][end]
            palindromic_split(start, end + 1)
            return result[start][end]

        palindromic_split(0, 0)
        print(f"Total hit found is {total_hit_count[0]}")
        return result[0][ls - 1]

    def minCut_timeout(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        result = [[-1] * len(s) for _ in range(len(s))]
        total_hit_count = [0]

        def palindromic_split(start, end):
            if start > end:
                return 0
            if result[start][end] != -1:
                this_string = str(s[start:end + 1])
                if this_string == this_string[::-1]:
                    result[start][end] = 0
                else:
                    min_k_count = float('inf')
                    for k in range(start, end):
                        k_res = palindromic_split(start, k) + 1 + palindromic_split(k + 1, end)
                        min_k_count = min(min_k_count, k_res)
                    result[start][end] = min_k_count
            else:
                total_hit_count[0] += 1
                print(f"Hit found for Result[{start}[{end}]")
            return result[start][end]

        result = [[0] * len(s) for _ in range(len(s))]
        res = palindromic_split(0, len(s) - 1)
        print(result)

        return res

    def minCut3(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        result = [[-1] * len(s) for _ in range(len(s))]

        def palindromic_split(start, end):
            if start > end:
                return 0
            if result[start][end] != -1:
                this_string = str(s[start:end + 1])
                print(f" *** Checking palindrome for {this_string} *** ")
                if this_string == this_string[::-1]:
                    result[start][end] = 0
                else:
                    min_k_count = float('inf')
                    for k in range(start, end):
                        k_res = palindromic_split(start, k) + 1 + palindromic_split(k + 1, end)
                        print(
                            f" !!! String => {s[start:end + 1]} invoking {s[start:k+1]} and {s[k+1:end+1]} !!! ==> gave {k_res}")
                        min_k_count = min(min_k_count, k_res)
                    print(f" !!! Result[{start}[{end}]=> !!! ==> gave {min_k_count}")
                    result[start][end] = min_k_count
            return result[start][end]

        result = [[0] * len(s) for _ in range(len(s))]
        res = palindromic_split(0, len(s) - 1)
        print(result)
        return res

    def minCutOld(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        ls = len(s)

        result = [[0] * len(s) for _ in range(len(s))]
        for j in range(ls):
            for i in range(ls):
                r, c = i, i + j
                if r == c:
                    result[r][c] = 0
                    continue

                if c < ls:
                    if s[r] == s[c]:
                        result[r][c] = result[r + 1][c - 1]
                    else:
                        result[r][c] = min(result[r][c - 1], result[r + 1][c]) + 1
        print(result)
        return result[0][ls - 1]


# https://leetcode.com/problems/edit-distance/description/
class MDSolution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        lw1, lw2 = len(word1), len(word2)

        if not word1 and not word2:
            return 0
        if word1 and not word2:
            return lw1
        if not word1 and word2:
            return lw2

        result = [[-1] * lw2 for _ in range(lw1)]

        def find_min(w1, w2):
            if w1 == -1:
                return w2 + 1
            elif w2 == -1:
                return w1 + 1

            if result[w1][w2] == -1:
                if word1[w1] == word2[w2]:
                    result[w1][w2] = find_min(w1 - 1, w2 - 1)
                else:
                    add_distance = find_min(w1 - 1, w2)
                    edit_distance = find_min(w1, w2 - 1)
                    sub_distance = find_min(w1 - 1, w2 - 1)
                    result[w1][w2] = 1 + min(add_distance, edit_distance, sub_distance)
            return result[w1][w2]

        find_min(lw1 - 1, lw2 - 1)
        return result[lw1 - 1][lw2 - 1]


# https://leetcode.com/problems/word-break-ii/description/
class WBSolution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        ls = len(s)
        if (not s and not wordDict) or (not s and wordDict):
            return None
        if s and len(wordDict) == 0:
            return []

        result = [[] for _ in range(ls)]
        from collections import namedtuple
        MergedDict = namedtuple('MergedDict', ('value', 'start', 'end'))

        for j in range(ls):
            for i in range(ls):
                if (i + j) < ls:
                    r, c = i, i + j
                    ss_in_dict = s[r:c + 1] in wordDict
                    if ss_in_dict:
                        (result[c]).append(MergedDict(value=s[r:c + 1], start=r, end=c))

        print(f"Step 1 complete ")
        end = result[-1]
        if not end:
            return []
        final_result = []

        def traverse_path(path, current_track):
            if not path:
                return None
            if path.start == 0:
                current_track.append(path.value)
                final_result.append(" ".join((current_track.copy())[::-1]))
                current_track.pop()
                return None

            child_paths = result[path.start - 1]
            if child_paths:
                for child_path in child_paths:
                    traverse_path(child_path, current_track + [path.value])
            else:
                return None

        for final in end:
            traverse_path(final, [])
        return final_result

    def wordBreakFailedRecursion(self, s, wordDict):
        ls = len(s)
        if (not s and not wordDict) or (not s and wordDict):
            return None
        if s and len(wordDict) == 0:
            return []

        result = [[-1] * ls for _ in range(ls)]
        from collections import namedtuple
        TreeResult = namedtuple('TreeResult', ('valid', 'kids'))

        def break_it(r, localDict):
            if r > ls - 1:
                return TreeResult(True, None)

            current_res = None
            for c in range(r, ls):
                if result[r][c] == -1:
                    cs = s[r: c + 1]
                    if cs in localDict:
                        # del localDict[localDict.index(cs)]
                        child_res = break_it(c + 1, localDict)
                        # localDict.append(cs)
                        print(f" ***{cs}*** in dictionary ==>  Result for String ***{s[c+1:]}*** ==> {child_res}")
                        if child_res and child_res.valid:
                            if child_res.kids:
                                merged_kids = []
                                for kid in child_res.kids:
                                    merged_kids.append([cs] + kid.copy())
                                result[r][c] = current_res = TreeResult(True, merged_kids)
                            else:
                                result[r][c] = current_res = TreeResult(True, [[cs]])
                        else:
                            result[r][c] = current_res = None  # TreeResult(False, None)
                        break
                    else:
                        result[r][c] = None  # TreeResult(False, None)
                    current_res = result[r][c]
                else:
                    current_res = result[r][c]
                    # print(f'Hit found ==> {current_res}')
            return current_res

        for _ in range(ls):
            break_it(0, wordDict)
        print(result)
        final_res = []
        for r in result[0]:
            if r and r != -1 and r.valid and r.kids:
                for kid in r.kids:
                    final_res.append(" ".join(kid))
        return final_res

    def wordBreakNew(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        ls = len(s)
        if (not s and not wordDict) or (not s and wordDict):
            return None
        if s and len(wordDict) == 0:
            return []

        result = [[None] * ls for _ in range(ls)]
        from collections import namedtuple
        MergedDict = namedtuple('MergedDict', ('value', 'start', 'end'))

        for j in range(ls):
            for i in range(ls):
                if (i + j) < ls:
                    r, c = i, i + j
                    left = result[r][c - 1] if c > 0 else None
                    bottom = result[r + 1][c] if (r + 1) < ls and c > 0 else None
                    ss_in_dict = s[r:c + 1] in wordDict
                    merged_result = []
                    if left and bottom:
                        # print(left)
                        # print(bottom)
                        for leftie in left:
                            for bottomie in bottom:
                                if leftie.end + 1 == bottomie.start:
                                    new_value = MergedDict(value=leftie.value + ' ' + bottomie.value,
                                                           start=leftie.start, end=bottomie.end)
                                    merged_result.append(new_value)
                        result[r][c - 1] = None
                        result[r + 1][c] = None
                        result[r][c] = merged_result
                    elif left or bottom:
                        # print(left if left else bottom)
                        merged_result = left if left else bottom
                        if left:
                            result[r][c - 1] = None
                        if bottom:
                            result[r + 1][c] = None
                    if ss_in_dict:
                        new_value = MergedDict(value=str(s[r:c + 1]), start=r, end=c)
                        merged_result.append(new_value)

                result[r][c] = merged_result if merged_result else None
        final_res = []
        if result[0][ls - 1]:
            for r in result[0][ls - 1]:
                if r.start == 0 and r.end == ls - 1:
                    final_res.append(r.value)
        return final_res

    def wordBreakOld(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        ls = len(s)
        result = [[-1] * ls for _ in range(ls)]

        def wordBreaks(r, c):
            if c > r or r > ls or c > ls:
                return None
            if result[r][c] == -1:
                cs = s[r:c + 1]
                if cs in wordDict:
                    down_result = wordBreaks(c + 1, c + 1)
                    if not down_result:
                        for dr in down_result:
                            pass
                else:
                    result[r][c] = wordBreaks(r, c + 1)

            return result[r][c]

        wordBreaks(0, 0)
        return result[0][0]


# https://leetcode.com/problems/create-maximum-number/description/
# https://leetcode.com/problems/create-maximum-number/discuss/77291/Share-my-Python-solution-with-explanation
class MNSolution(object):
    def maxNumber(self, nums1, nums2, k):
        n, m = len(nums1), len(nums2)
        if n > m:
            nums1, nums2 = nums2, nums1
            n, m = len(nums1), len(nums2)
        ret = [0] * k

        for i in range(0, k + 1):
            j = k - i
            if i > n or j > m: continue
            left = self.get_max_numbers(nums1, i)
            right = self.get_max_numbers(nums2, j)
            num = self.merge_number(left, right)
            ret = max(num, ret)
            print(ret)
        return ret

    def get_max_numbers(self, nums1, k):
        import operator
        current_start, current_num = 0, 1
        ln = len(nums1)
        result = []
        if k <= ln:
            while current_start < ln and current_num < (k + 1):
                current_end = -(int(k - current_num)) if (k - current_num) > 0 else None
                max_index, max_value = max(enumerate(nums1[current_start:current_end]), key=operator.itemgetter(1))
                current_start = current_start + max_index + 1
                current_num += 1
                result.append(max_value)
        return result

    def merge_number(self, nums1, nums2):
        if not nums1 and not nums2:
            return nums1
        if nums1 and not nums2:
            return nums1
        if not nums1 and nums2:
            return nums2
        result = []
        # if len(nums2) > len(nums1):
        #     nums1, nums2 = nums2, nums1

        while nums1 and nums2:
            n1, n2 = nums1[0], nums2[0]
            if n1 > n2:
                result.append(nums1.pop(0))
            elif n1 < n2:
                result.append(nums2.pop(0))
            elif n1 == n2:
                nn1 = nums1[1] if len(nums1) > 1 else None
                nn2 = nums2[1] if len(nums2) > 1 else None
                if nn1 is not None and nn2 is not None:
                    if nn1 == nn2 or nn1 > nn2:
                        result.append(nums1.pop(0))
                    else:
                        result.append(nums2.pop(0))
                elif nn1 is not None and nn2 is None:
                    result.append(nums1.pop(0))
                elif nn1 is None and nn2 is not None:
                    result.append(nums2.pop(0))
                else:
                    result.append(nums1.pop(0))

        return (result + nums1) if nums1 else result + nums2

    def maxNumber_curr(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[int]
        """
        n, m = len(nums1), len(nums2)
        ret = [0] * k
        for i in range(0, k + 1):
            j = k - i
            if i > n or j > m: continue
            left = self.maxSingleNumber(nums1, i)
            right = self.maxSingleNumber(nums2, j)
            num = self.mergeMax(left, right)
            ret = max(num, ret)
        return ret

    def mergeMax(self, nums1, nums2):
        ans = []
        while nums1 or nums2:
            if nums1 > nums2:
                ans += nums1[0],
                nums1 = nums1[1:]
            else:
                ans += nums2[0],
                nums2 = nums2[1:]
        return ans

    def maxSingleNumber(self, nums, selects):
        n = len(nums)
        ret = [-1]
        if selects > n: return ret
        while selects > 0:
            start = ret[-1] + 1  # search start
            end = n - selects + 1  # search end
            ret.append(max(range(start, end), key=nums.__getitem__))
            selects -= 1
        ret = [nums[item] for item in ret[1:]]
        return ret


if __name__ == "__main__":
    obj = MNSolution()
    start = time.clock()
    print(time.clock() - start)

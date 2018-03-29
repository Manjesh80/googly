from ..dynamic import *


# https://leetcode.com/explore/interview/card/google/64/dynamic-programming-4/334/
# Decode Ways
# pytest EPI\tests\test_dynamic_leet_google.py::test_numDecodings
def test_numDecodings():
    obj = NDSolution()
    assert obj.numDecodings(s='10') == 1
    assert obj.numDecodings(s='12') == 2
    assert obj.numDecodings(s='123') == 3
    assert obj.numDecodings(s='999') == 1
    assert obj.numDecodings(s='0') == 0
    assert obj.numDecodings(s='01') == 0
    assert obj.numDecodings(s='903') == 0
    assert obj.numDecodings(s='103') == 1
    assert obj.numDecodings(s='100') == 0
    assert obj.numDecodings(s='110') == 1
    assert obj.numDecodings(s='20') == 1
    assert obj.numDecodings(s='90') == 0


# https://leetcode.com/problems/decode-ways-ii/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_numDecodings_2
def test_numDecodings_2():
    obj = ND2Solution()
    assert obj.numDecodings(s='*1*1*0') == 404
    assert obj.numDecodings(s='*10') == 9
    assert obj.numDecodings(s='12') == 2
    assert obj.numDecodings(s='123') == 3
    assert obj.numDecodings(s="*10*1") == 99


# https://leetcode.com/problems/regular-expression-matching/description/
# Expression matching
# pytest EPI\tests\test_dynamic_leet_google.py::test_ExpMatchSolution
def test_ExpMatchSolution():
    obj = ExpMatchSolution()
    assert obj.isMatch("aa", "a") == False
    assert obj.isMatch("aa", "aa") == True
    assert obj.isMatch("aaa", "aa") == False
    assert obj.isMatch("aa", "a*") == True
    assert obj.isMatch("aa", ".*") == True
    assert obj.isMatch("ab", ".*") == True
    assert obj.isMatch("aab", "c*a*b") == True
    assert obj.isMatch("ab", ".*c") == False
    assert obj.isMatch("aaa", "a*a") == True
    assert obj.isMatch("aaa", "ab*a*c*a") == True
    assert obj.isMatch("abcd", "d*") == False
    assert obj.isMatch("aaa", "ab*a") == False


# https://leetcode.com/problems/maximal-rectangle/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_maximalRectangle
def test_maximalRectangle():
    obj = MRSolution()
    assert obj.maximalRectangle([[1, 1], [1, 1]]) == 4
    assert obj.maximalRectangle([[1, 1], [1, 0]]) == 0
    assert obj.maximalRectangle([[1, 1, 1], [1, 1, 1]]) == 6
    assert obj.maximalRectangle([[1, 0, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0]]) == 6


# https://leetcode.com/problems/wildcard-matching/description/
# Wildcard matching
# pytest EPI\tests\test_dynamic_leet_google.py::test_WCMatchSolution
def test_WCMatchSolution():
    obj = WCSolution()
    assert obj.isMatch("aa", "a") is False
    assert obj.isMatch("aa", "aa") is True
    assert obj.isMatch("aaa", "aa") is False
    assert obj.isMatch("aa", "*") is True
    assert obj.isMatch("aa", "a*") is True
    assert obj.isMatch("ab", "?*") is True
    assert obj.isMatch("aab", "c*a*b") is False
    assert obj.isMatch("ho", "**ho") is True


# https://leetcode.com/problems/longest-valid-parentheses/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_LVPSolution
def test_LVPSolution():
    obj = LVPSolution()
    assert obj.longestValidParentheses(")()") == 2
    assert obj.longestValidParentheses(")()(") == 2
    # assert obj.longestValidParentheses(")()())") == 4
    assert obj.longestValidParentheses(")(") == 0
    assert obj.longestValidParentheses(")") == 0
    assert obj.longestValidParentheses("(") == 0
    assert obj.longestValidParentheses("") == 0
    assert obj.longestValidParentheses("(()") == 2


# https://leetcode.com/problems/interleaving-string/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_ILSolution
def test_ILSolution():
    obj = ILSolution()
    assert obj.isInterleave("aabcc", "dbbca", "aadbbcbcac") is True
    assert obj.isInterleave("YY", "YZ", "YYZY") is True
    assert obj.isInterleave(
        "baababbabbababbaaababbbbbbbbbbbaabaabaaaabaaabbaaabaaaababaabaaabaabbbbaabbaabaabbbbabbbababbaaaabab",
        "aababaaabbbababababaabbbababaababbababbbbabbbbbababbbabaaaaabaaabbabbaaabbababbaaaababaababbbbabbbbb",
        "babbabbabbababbaaababbbbaababbaabbbbabbbbbaaabbabaababaabaaabaabbbaaaabbabbaaaaabbabbaabaaaabbbbababbbababbabaabababbababaaaaaabbababaaabbaabbbbaaaaabbbaaabbbabbbbaaabaababbaabababbbbababbaaabbbabbbab") is True


# https://leetcode.com/problems/palindrome-partitioning-ii/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_MCSolution
def test_MCSolution():
    obj = MCSolution()
    assert obj.minCut('abc') == 2
    assert obj.minCut('aaa') == 0
    assert obj.minCut('aba') == 0
    assert obj.minCut('daab') == 2
    assert obj.minCut('aab') == 1


# https://leetcode.com/problems/word-break-ii/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_WBSolution
def test_WBSolution():
    obj = WBSolution()
    assert obj.wordBreak("catdog", ["cat", "dog"])
    assert obj.wordBreak("catsanddog", ["cat", "cats", "and", "sand", "dog"])
    # print(obj.wordBreak("catdog", ["cat", "dog"]))
    # print(obj.wordBreak("catsanddog", ["cat", "cats", "and", "sand", "dog"]))
    # print(obj.wordBreak("catsanddog", ["cat", "cats", "and", "sand", "dog"]))
    # print(obj.wordBreak("catsanddog", ["cat", "cats", "and", "sand", "dog"]))
    # print(obj.wordBreak("sanddog", ["cat", "cats", "and", "sand", "dog"]))
    # print(obj.wordBreak("catsand", ["catx", "sand"]))
    # print(obj.wordBreak("AB", ["A", "B"]))
    # print(obj.wordBreak("cats", ["cat", "cats"]))
    # print(obj.wordBreak("aaaaaaa", ["aaaa", "aaa"]))
    # print(obj.wordBreak("bb", ["b", "b"]))
    # print(obj.wordBreak(
    #     # "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    #     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    #     ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"]))


# https://leetcode.com/problems/create-maximum-number/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_MNSolution
def test_MNSolution():
    obj = MNSolution()
    assert obj.maxNumber([2, 1, 7, 8, 0, 1, 7, 3, 5, 8, 9, 0, 0, 7, 0, 2, 2, 7, 3, 5, 5],
                         [2, 6, 2, 0, 1, 0, 5, 4, 5, 5, 3, 3, 3, 4], 35) == [2, 6, 2, 2, 1, 7, 8, 0, 1, 7, 3, 5, 8, 9,
                                                                             0, 1, 0, 5, 4, 5, 5, 3, 3, 3, 4, 0, 0, 7,
                                                                             0, 2, 2, 7, 3, 5, 5]

    assert obj.maxNumber([5, 0, 2, 1, 0, 1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3],
                         [7, 6, 7, 1, 0, 1, 0, 5, 6, 0, 5, 0], 31) == [7, 6, 7, 5, 1, 0, 2, 1, 0, 1, 0, 5, 6, 0, 5, 0,
                                                                       1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3, 0]

    assert obj.merge_number([2, 6, 2, 0, 1, 0, 5, 4, 5, 5, 3, 3, 3, 4],
                            [2, 1, 7, 8, 0, 1, 7, 3, 5, 8, 9, 0, 0, 7, 0, 2, 2, 7, 3, 5, 5]) == [2, 6, 2, 2, 1, 7, 8, 0,
                                                                                                 1, 7, 3, 5, 8, 9,
                                                                                                 0, 1, 0, 5, 4, 5, 5, 3,
                                                                                                 3, 3, 4, 0, 0, 7,
                                                                                                 0, 2, 2, 7, 3, 5, 5]
    # print(obj.maxNumber(nums1=[3, 4, 6, 5], nums2=[9, 1, 2, 5, 8, 3], k=5))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=0))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=1))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=2))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=3))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=4))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=5))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=6))
    # print(obj.get_max_numbers([9, 1, 2, 5, 8, 3], k=7))
    #
    # print(obj.merge_number([6, 7], [6, 0, 4]))
    # print(obj.merge_number([7, 6, 7, 1, 0, 1, 0, 5, 6, 0, 5, 0],
    #                        [5, 0, 2, 1, 0, 1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3]))

    # print(obj.maxNumber([6, 7], [6, 0, 4], 5))
    # print(obj.maxNumber([5, 0, 2, 1, 0, 1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3],
    #                     [7, 6, 7, 1, 0, 1, 0, 5, 6, 0, 5, 0], 31))

    # print(obj.maxNumber([2, 1, 7, 8, 0, 1, 7, 3, 5, 8, 9, 0, 0, 7, 0, 2, 2, 7, 3, 5, 5],
    #                     [2, 6, 2, 0, 1, 0, 5, 4, 5, 5, 3, 3, 3, 4], 35))

    # res = obj.merge_number([2, 6, 2, 0, 1, 0, 5, 4, 5, 5, 3, 3, 3, 4],
    #                        [2, 1, 7, 8, 0, 1, 7, 3, 5, 8, 9, 0, 0, 7, 0, 2, 2, 7, 3, 5, 5])
    #
    # print(res == [2, 6, 2, 2, 1, 7, 8, 0, 1, 7, 3, 5, 8, 9, 0, 1, 0, 5, 4, 5, 5, 3, 3, 3, 4, 0, 0, 7, 0, 2, 2, 7, 3, 5,
    #               5])

    # res = obj.merge_number([7, 6, 7, 1, 0, 1, 0, 5, 6, 0, 5, 0],
    #                        [5, 0, 2, 1, 0, 1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3])

    # res = obj.merge_number([1, 0, 1, 0, 5],
    #                        [1, 0, 1, 0, 3])
    # print(res)
    # print(res == [7, 6, 7, 5, 1, 0, 2, 1, 0, 1, 0, 5, 6, 0, 5, 0, 1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3, 0])
    #                     [5, 1, 0, 2, 1, 0, 1, 0, 3, 0, 1, 0, 5]

    # print(obj.maxNumber([5, 0, 2, 1, 0, 1, 0, 3, 9, 1, 2, 8, 0, 9, 8, 1, 4, 7, 3],
    #                     [7, 6, 7, 1, 0, 1, 0, 5, 6, 0, 5, 0], 31))

    # print(obj.wordBreak("catdog", ["cat", "dog"]))


# https://leetcode.com/problems/frog-jump/description/
# pytest EPI\tests\test_dynamic_leet_google.py::test_CCSolution
def test_CCSolution():
    obj = CCSolution()
    assert obj.canCross([0, 1, 3, 5, 6, 8, 12, 17])
    assert obj.canCross([0, 1, 2, 3, 4, 8, 9, 11]) is False
    assert obj.canCross([0, 2]) is False
    assert obj.canCross([0, 1, 3, 6, 10, 15, 16, 21])

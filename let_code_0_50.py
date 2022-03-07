# -*- coding: utf-8 -*-
import sys
import heapq
import collections
from typing import List, Optional


class Solution1:
    """1. 两数之和
    https://leetcode-cn.com/problems/two-sum/

    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
    你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
    你可以按任意顺序返回答案。

    输入：nums = [2,7,11,15], target = 9
    输出：[0,1]
    解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

    输入：nums = [3,2,4], target = 6
    输出：[1,2]

    输入：nums = [3,3], target = 6
    输出：[0,1]

    """

    @staticmethod
    def two_sum(nums: List[int], target: int) -> List[int]:
        hash_table = dict()
        for i, num in enumerate(nums):
            if target - num in hash_table:
                return [hash_table[target - num], i]
            hash_table[nums[i]] = i
        return []


class Solution2:
    """
    2. 两数相加
    https://leetcode-cn.com/problems/add-two-numbers/
    给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
    请你将两个数相加，并以相同形式返回一个表示和的链表。
    你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
    输入：l1 = [2,4,3], l2 = [5,6,4]
    输出：[7,0,8]
    解释：342 + 465 = 807

    输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
    输出：[8,9,9,9,0,0,0,1]
    """

    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None

    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        root = self.ListNode(0)
        cursor = root
        carry = 0
        while l1 or l2 or carry != 0:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0

            sum_val = l1_val + l2_val + carry
            carry = int(sum_val / 10)

            sum_node = self.ListNode(sum_val % 10)

            cursor.next = sum_node
            cursor = sum_node

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return root.next


class Solution3:
    """
    https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
    3. 无重复字符的最长子串
    给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

    输入: s = "abcabcbb"
    输出: 3
    解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

    输入: s = "pwwkew"
    输出: 3
    解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
         请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。


    """

    @staticmethod
    def length_of_longest_sub_string(s: str) -> int:
        s_length = len(s)
        s_list = []
        max_length = 0
        for i in range(s_length):
            while s[i] in s_list:
                s_list.remove(s_list[0])
            else:
                s_list.append(s[i])
                max_length = max(max_length, len(s_list))
        return max_length


class Solution4:
    """
    4. 寻找两个正序数组的中位数
    https://leetcode-cn.com/problems/median-of-two-sorted-arrays/
    输入：nums1 = [1,3], nums2 = [2]
    输出：2.00000
    解释：合并数组 = [1,2,3] ，中位数 2

    输入：nums1 = [1,2], nums2 = [3,4]
    输出：2.50000
    解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5

    """

    @staticmethod
    def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
        def get_kth_element(k):
            """
            - 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
            - 这里的 "/" 表示整除
            - nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
            - nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
            - 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
            - 这样 pivot 本身最大也只能是第 k-1 小的元素
            - 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
            - 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
            - 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
            """

            index1, index2 = 0, 0
            while True:
                # 特殊情况
                if index1 == m:
                    return nums2[index2 + k - 1]
                if index2 == n:
                    return nums1[index1 + k - 1]
                if k == 1:
                    return min(nums1[index1], nums2[index2])

                # 正常情况
                new_index1 = min(index1 + k // 2 - 1, m - 1)
                new_index2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[new_index1], nums2[new_index2]
                if pivot1 <= pivot2:
                    k -= new_index1 - index1 + 1
                    index1 = new_index1 + 1
                else:
                    k -= new_index2 - index2 + 1
                    index2 = new_index2 + 1

        m, n = len(nums1), len(nums2)
        total_length = m + n
        if total_length % 2 == 1:
            return get_kth_element((total_length + 1) // 2)
        else:
            return (get_kth_element(total_length // 2) + get_kth_element(total_length // 2 + 1)) / 2


class Solution5:
    """
    5. 最长回文子串
    https://leetcode-cn.com/problems/longest-palindromic-substring/

    输入：s = "babad"
    输出："bab"
    解释："aba" 同样是符合题意的答案。

    输入：s = "cbbd"
    输出："bb"

    """

    @staticmethod
    def longest_palindrome(s: str) -> str:
        result = s[0]
        if s == s[::-1]:
            return s
        for i in range(2, len(s)):
            for j in range(0, len(s) - i + 1):
                sub_s = s[j:j + i]
                if sub_s == sub_s[::-1] and len(sub_s) > len(result):
                    result = sub_s
        return result


class Solution6:
    """
    6. Z 字形变换
    https://leetcode-cn.com/problems/zigzag-conversion/
    将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
    输入：s = "PAYPALISHIRING", numRows = 3
    输出："PAHNAPLSIIGYIR"

    输入：s = "PAYPALISHIRING", numRows = 4
    输出："PINALSIGYAHRPI"
    解释：
    P     I    N
    A   L S  I G
    Y A   H R
    P     I

    """

    @staticmethod
    def convert(s: str, num_rows: int) -> str:
        if num_rows < 2: return s
        res = ["" for _ in range(num_rows)]
        i, flag = 0, -1
        for c in s:
            res[i] += c
            if i == 0 or i == num_rows - 1:
                flag = -flag
            i += flag
        return "".join(res)


class Solution7:
    """
    7. 整数反转
    https://leetcode-cn.com/problems/reverse-integer/
    给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
    如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。

    输入：x = 123
    输出：321

    输入：x = -123
    输出：-321

    """

    @staticmethod
    def reverse(x: int) -> int:
        res = 0
        zz = 10 if x > 0 else -10
        while x != 0:
            tmp = x % zz
            if res > 214748364 or (res == 214748364 and tmp > 7):
                return 0
            if res < -214748364 or (res == -214748364 and tmp < -8):
                return 0
            res = res * 10 + tmp
            x = int(x / 10)
        return res

    @staticmethod
    def reverse2(x: int) -> int:
        INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1

        rev = 0
        while x != 0:
            # INT_MIN 也是一个负数，不能写成 rev < INT_MIN // 10
            if rev < INT_MIN // 10 + 1 or rev > INT_MAX // 10:
                return 0
            digit = x % 10
            # Python3 的取模运算在 x 为负数时也会返回 [0, 9) 以内的结果，因此这里需要进行特殊判断
            if x < 0 and digit > 0:
                digit -= 10

            # 同理，Python3 的整数除法在 x 为负数时会向下（更小的负数）取整，因此不能写成 x //= 10
            x = (x - digit) // 10
            rev = rev * 10 + digit

        return rev


class Solution8:
    """
    8. 字符串转换整数 (atoi)
    https://leetcode-cn.com/problems/string-to-integer-atoi/
    请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

    函数 myAtoi(string s) 的算法如下：

    读入字符串并丢弃无用的前导空格
    检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
    读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
    将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
    如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
    返回整数作为最终结果。

    输入：s = "42"
    输出：42
    解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
    第 1 步："42"（当前没有读入字符，因为没有前导空格）
    第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
    第 3 步："42"（读入 "42"）
    解析得到整数 42 。
    由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。

    输入：s = "4193 with words"
    输出：4193
    解释：
    第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）^
    第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
    第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
    解析得到整数 4193 。
    由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。



    """
    INT_MAX = 2 ** 31 - 1
    INT_MIN = -2 ** 31

    class Automaton:
        def __init__(self):
            self.state = 'start'
            self.sign = 1
            self.ans = 0
            self.table = {
                'start': ['start', 'signed', 'in_number', 'end'],
                'signed': ['end', 'end', 'in_number', 'end'],
                'in_number': ['end', 'end', 'in_number', 'end'],
                'end': ['end', 'end', 'end', 'end'],
            }

        @staticmethod
        def get_col(c):
            if c.isspace():
                return 0
            if c == '+' or c == '-':
                return 1
            if c.isdigit():
                return 2
            return 3

        def get(self, c):
            self.state = self.table[self.state][self.get_col(c)]
            if self.state == 'in_number':
                self.ans = self.ans * 10 + int(c)
                self.ans = min(self.ans, Solution8.INT_MAX) if self.sign == 1 else min(self.ans, -Solution8.INT_MIN)
            elif self.state == 'signed':
                self.sign = 1 if c == '+' else -1

    def my_atoi(self, str: str) -> int:
        automaton = self.Automaton()
        for c in str:
            automaton.get(c)
        return automaton.sign * automaton.ans


class Solution9:
    """
    9. 回文数
    https://leetcode-cn.com/problems/palindrome-number/
    给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
    回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
    例如，121 是回文，而 123 不是。

    输入：x = 121
    输出：true

    输入：x = -121
    输出：false
    解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。

    """

    # 方法一: 将int转化成str类型: 双向队列
    # 复杂度: O(n^2) [每次pop(0)都是O(n)..比较费时]
    @staticmethod
    def is_palindrome_01(x: int) -> bool:
        lst = list(str(x))
        while len(lst) > 1:
            if lst.pop(0) != lst.pop():
                return False
        return True

    # 方法二: 将int转化成str类型: 双指针 (指针的性能一直都挺高的)
    # 复杂度: O(n)
    @staticmethod
    def is_palindrome_02(x: int) -> bool:
        lst = list(str(x))
        L, R = 0, len(lst) - 1
        while L <= R:
            if lst[L] != lst[R]:
                return False
            L += 1
            R -= 1
        return True

    # 方法三: 进阶:不将整数转为字符串来解决: 使用log来计算x的位数
    # 复杂度: O(n)
    @staticmethod
    def is_palindrome_03(x: int) -> bool:
        """
        模仿上面字符串的方法:分别取'第一位的数'与'第二位的数'对比
                        (弊端是:频繁计算,导致速度变慢)(下面的方法三只反转一半数字,可以提高性能)
        """
        if x < 0:
            return False
        elif x == 0:
            return True
        else:
            import math
            length = int(math.log(x, 10)) + 1
            L = length - 1
            print("l = ", L)
            for i in range(length // 2):
                if x // 10 ** L != x % 10:
                    return False
                x = (x % 10 ** L) // 10
                L -= 2
            return True

    # 方法四: 进阶:不将整数转为字符串来解决: 使用log来计算x的位数
    # 复杂度: O(n)
    @staticmethod
    def is_palindrome_04(x: int) -> bool:
        """
        只反转后面一半的数字!!(节省一半的时间)
        """
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        elif x == 0:
            return True
        else:
            import math
            length = int(math.log(x, 10)) + 1
            reverse_x = 0
            for i in range(length // 2):
                remainder = x % 10
                x = x // 10
                reverse_x = reverse_x * 10 + remainder
            # 当x为奇数时, 只要满足 reverse_x == x//10 即可
            if reverse_x == x or reverse_x == x // 10:
                return True
            else:
                return False

    # 方法五: 进阶:不将整数转为字符串来解决: 不使用log函数
    # 复杂度: O(n)
    @staticmethod
    def is_palindrome_05(x: int) -> bool:
        """
        只反转后面一半的数字!!(节省一半的时间)
        """
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        elif x == 0:
            return True
        else:
            reverse_x = 0
            while x > reverse_x:
                remainder = x % 10
                reverse_x = reverse_x * 10 + remainder
                x = x // 10
            # 当x为奇数时, 只要满足 reverse_x//10 == x 即可
            if reverse_x == x or reverse_x // 10 == x:
                return True
            else:
                return False


class Solution10:
    """
    10. 正则表达式匹配
    https://leetcode-cn.com/problems/regular-expression-matching/
    给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
    '.' 匹配任意单个字符
    '*' 匹配零个或多个前面的那一个元素

    输入：s = "aa", p = "a"
    输出：false
    解释："a" 无法匹配 "aa" 整个字符串。

    输入：s = "aa", p = "a*"
    输出：true
    解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。

    输入：s = "ab", p = ".*"
    输出：true
    解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。

    """

    @staticmethod
    def is_match(text, pattern):
        d = {}

        def dfs(s, p):
            # 加个记忆化，不然纯递归会超时的
            if (s, p) in d:
                return d[s, p]
            if not p:
                return not s
            # 检查s[0]和p[0]是否匹配
            is_first_match = len(s) > 0 and p[0] in (s[0], ".")
            # 检查模式串的下个字符是否为"*
            try_match_any = len(p) >= 2 and p[1] == "*"
            if try_match_any:
                # 如果包含"*"，可以忽略掉当前字符+*
                # 也可以忽略掉字符串中的当前字符(如果能匹配上)
                d[s, p] = dfs(s, p[2:]) or (is_first_match and dfs(s[1:], p))
            else:
                # 单个字符匹配的情况
                d[s, p] = is_first_match and dfs(s[1:], p[1:])
            return d[s, p]

        return dfs(text, pattern)

    @staticmethod
    def is_match_1(s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i: int, j: int) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]


class Solution11:
    """
    11. 盛最多水的容器
    https://leetcode-cn.com/problems/container-with-most-water/
    给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
    找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
    返回容器可以储存的最大水量。

    输入：[1,8,6,2,5,4,8,3,7]
    输出：49
    解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。


    """

    @staticmethod
    def max_area(height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            ans = max(ans, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return ans


class Solution12:
    """
    12. 整数转罗马数字
    https://leetcode-cn.com/problems/integer-to-roman/
    罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
    字符          数值
    I             1
    V             5
    X             10
    L             50
    C             100
    D             500
    M             1000

    例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
    通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
    I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
    X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
    C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
    给你一个整数，将其转为罗马数字。

    输入: num = 3
    输出: "III"

    输入: num = 4
    输出: "IV"

    输入: num = 1994
    输出: "MCMXCIV"
    解释: M = 1000, CM = 900, XC = 90, IV = 4.

    """
    VALUE_SYMBOLS = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I")
    ]

    def int_to_roman(self, num: int) -> str:
        roman = list()
        for value, symbol in self.VALUE_SYMBOLS:
            while num >= value:
                num -= value
                roman.append(symbol)
            if num == 0:
                break
        return "".join(roman)


class Solution13:
    """
    13. 罗马数字转整数
    https://leetcode-cn.com/problems/roman-to-integer/
    罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。
    例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

    通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

    I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
    X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
    C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
    给定一个罗马数字，将其转换成整数。

    输入: s = "III"
    输出: 3

    输入: s = "MCMXCIV"
    输出: 1994
    解释: M = 1000, CM = 900, XC = 90, IV = 4.

    """
    VALUE_SYMBOLS = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I")
    ]

    def roman_to_int(self, s: str) -> int:
        res = 0
        for value, symbol in self.VALUE_SYMBOLS:
            while str(s).startswith(symbol):
                res += value
                s = s[len(symbol):]
            if len(s) == 0:
                break
        return res


class Solution14:
    """
    14. 最长公共前缀
    https://leetcode-cn.com/problems/longest-common-prefix/

    编写一个函数来查找字符串数组中的最长公共前缀。
    如果不存在公共前缀，返回空字符串 ""。

    输入：strs = ["flower","flow","flight"]
    输出："fl"


    """

    @staticmethod
    def longest_common_prefix(str_list: List[str]) -> str:
        def is_common_prefix(size):
            str_0, count = str_list[0][:size], len(str_list)
            return all(str_list[i][:size] == str_0 for i in range(1, count))

        if not str_list:
            return ""

        min_length = min([len(x) for x in str_list])
        low, high = 0, min_length
        while low < high:
            mid = (high - low + 1) // 2 + low
            if is_common_prefix(mid):
                low = mid
            else:
                high = mid - 1
        return str_list[0][:low]


class Solution15:
    """
    15. 三数之和
    https://leetcode-cn.com/problems/3sum/
    给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
    注意：答案中不可以包含重复的三元组。

    输入：nums = [-1,0,1,2,-1,-4]
    输出：[[-1,-1,2],[-1,0,1]]


    """

    @staticmethod
    def three_sum(nums):
        n = len(nums)
        result = []
        if not nums or n < 3:
            return result
        nums.sort()
        for i in range(n):
            if nums[i] > 0:
                return result
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l = i + 1
            r = n - 1
            while l < r:
                if nums[i] + nums[l] + nums[r] == 0:
                    result.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif nums[i] + nums[l] + nums[r] > 0:
                    r -= 1
                else:
                    l += 1
        return result


class Solution16:
    """
    16. 最接近的三数之和
    https://leetcode-cn.com/problems/3sum-closest/
    给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。
    返回这三个数的和。
    假定每组输入只存在恰好一个解。

    输入：nums = [-1,2,1,-4], target = 1
    输出：2
    解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2)

    """

    @staticmethod
    def three_sum_closest(nums, target):
        n = len(nums)
        flag = sys.maxsize
        if not nums or n < 3:
            return
        nums.sort()
        result = None
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l = i + 1
            r = n - 1
            while l < r:
                gap = abs(nums[i] + nums[l] + nums[r] - target)
                if gap == 0:
                    return nums[i] + nums[l] + nums[r]
                if gap < flag:
                    result = nums[i] + nums[l] + nums[r]
                    flag = gap
                elif nums[i] + nums[l] + nums[r] > target:
                    r -= 1
                else:
                    l += 1
        return result


class Solution17:
    """
    17. 电话号码的字母组合
    https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/

    给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
    给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

    输入：digits = "23"
    输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

    输入：digits = "2"
    输出：["a","b","c"]

    """

    @staticmethod
    def letter_combinations(digits: str):
        phone = {"2": ["a", "b", "c"],
                 "3": ["d", "e", "f"],
                 "4": ["g", "h", "i"],
                 "5": ["j", "k", "l"],
                 "6": ["m", "n", "o"],
                 "7": ["p", "q", "r", "s"],
                 "8": ["t", "u", "v"],
                 "9": ["w", "x", "y", "z"]
                 }
        queue = [""]
        for digit in digits:
            for i in range(len(queue)):
                tmp = queue.pop(0)
                for letter in phone[digit]:
                    queue.append(tmp + letter)
        if len(queue) == 1 and not queue[0]:
            return []
        return queue


class Solution18:
    """
    18. 四数之和
    https://leetcode-cn.com/problems/4sum/
    输入：nums = [1,0,-1,0,-2,2], target = 0
    输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

    输入：nums = [2,2,2,2,2], target = 8
    输出：[[2,2,2,2]]

    """

    @staticmethod
    def four_sum(nums: List[int], target: int) -> List[List[int]]:
        if not nums or len(nums) < 4:
            return []
        nums.sort()
        res = []
        for a in range(len(nums) - 3):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            for b in range(a + 1, len(nums) - 2):
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                c = b + 1
                d = len(nums) - 1
                while c < d:
                    sum = nums[a] + nums[b] + nums[c] + nums[d]
                    if sum == target:
                        res.append([nums[a], nums[b], nums[c], nums[d]])
                        while c < d and nums[c] == nums[c + 1]:
                            c += 1
                        while c < d and nums[d] == nums[d - 1]:
                            d -= 1
                        c += 1
                        d -= 1
                    elif sum < target:
                        c += 1
                    else:
                        d -= 1
        return res


class Solution19:
    """
    19. 删除链表的倒数第 N 个结点
    https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
    给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
    输入：head = [1,2,3,4,5], n = 2
    输出：[1,2,3,5]

    输入：head = [1,2], n = 1
    输出：[1]
    """

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def remove_nth_from_end(self, head: ListNode, n: int) -> ListNode:
        pre = self.ListNode(-1, head)
        slow = pre
        fast = pre
        while n > 0:
            fast = fast.next
            n -= 1
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return pre.next


class Solution20:
    """
    20. 有效的括号
    https://leetcode-cn.com/problems/valid-parentheses/
    给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
    有效字符串需满足：
    左括号必须用相同类型的右括号闭合。
    左括号必须以正确的顺序闭合。

    输入：s = "()[]{}"
    输出：true

    输入：s = "([)]"
    输出：false
    """

    @staticmethod
    def is_valid(s: str) -> bool:
        dic = {'{': '}', '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic:
                stack.append(c)
            elif dic.get(stack.pop()) != c:
                return False
        return len(stack) == 1


class Solution21:
    """
    21. 合并两个有序链表
    https://leetcode-cn.com/problems/merge-two-sorted-lists/
    将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
    输入：l1 = [1,2,4], l2 = [1,3,4]
    输出：[1,1,2,3,4,4]

    输入：l1 = [], l2 = [0]
    输出：[0]

    """

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def merge_two_lists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head1 = list1
        head2 = list2
        cur = self.ListNode()
        head = cur
        while head1 and head2:
            if head1.val < head2.val:
                cur.next = head1
                cur = head1
                head1 = head1.next
            else:
                cur.next = head2
                cur = head2
                head2 = head2.next
        if head1:
            cur.next = head1
        if head2:
            cur.next = head2
        return head.next


class Solution22:
    """
    22. 括号生成
    https://leetcode-cn.com/problems/generate-parentheses/
    数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

    输入：n = 3
    输出：["((()))","(()())","(())()","()(())","()()()"]

    """

    @staticmethod
    def generate_parenthesis(n: int) -> List[str]:

        res = []
        cur_str = ''

        def dfs(cur_str, left, right):
            if left == 0 and right == 0:
                res.append(cur_str)
                return
            if left > 0:
                dfs(cur_str + '(', left - 1, right)
            if right > 0 and right > left:
                dfs(cur_str + ')', left, right - 1)

        dfs(cur_str, n, n)
        return res


class Solution23:
    """
    23. 合并K个升序链表
    https://leetcode-cn.com/problems/merge-k-sorted-lists/
    给你一个链表数组，每个链表都已经按升序排列。
    请你将所有链表合并到一个升序链表中，返回合并后的链表。

    输入：lists = [[1,4,5],[1,3,4],[2,6]]
    输出：[1,1,2,3,4,4,5,6]
    解释：链表数组如下：
    [
      1->4->5,
      1->3->4,
      2->6
    ]
    将它们合并到一个有序链表中得到。
    1->1->2->3->4->4->5->6

    """

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def merge_k_lists(self, lists: List[ListNode]) -> ListNode:
        min_heap = []
        for i in lists:
            while i:
                heapq.heappush(min_heap, i.val)  # 把i中的数据逐个加到堆中
                i = i.next
        dummy = self.ListNode(0)  # 构造虚节点
        p = dummy
        while min_heap:
            p.next = self.ListNode(heapq.heappop(min_heap))  # 依次弹出最小堆的数据
            p = p.next
        return dummy.next


class Solution24:
    """
    24. 两两交换链表中的节点
    https://leetcode-cn.com/problems/swap-nodes-in-pairs/
    给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
    输入：head = [1,2,3,4]
    输出：[2,1,4,3]
    """

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def swap_pairs(self, head: ListNode) -> ListNode:
        dummy_head = self.ListNode(0)
        dummy_head.next = head
        temp = dummy_head
        while temp.next and temp.next.next:
            node1 = temp.next
            node2 = temp.next.next
            temp.next = node2
            node1.next = node2.next
            node2.next = node1
            temp = node1
        return dummy_head.next


class Solution25:
    """
    25. K 个一组翻转链表
    https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
    给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
    k 是一个正整数，它的值小于或等于链表的长度。
    如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

    输入：head = [1,2,3,4,5], k = 2
    输出：[2,1,4,3,5]

    输入：head = [1,2,3,4,5], k = 3
    输出：[3,2,1,4,5]

    输入：head = [1,2,3,4,5], k = 1
    输出：[1,2,3,4,5]

    """

    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def reverse_k_group(self, head: ListNode, k: int) -> ListNode:
        dummy = self.ListNode(0)
        p = dummy
        while True:
            count = k
            stack = []
            tmp = head
            while count and tmp:
                stack.append(tmp)
                tmp = tmp.next
                count -= 1
            # 注意,目前tmp所在k+1位置
            # 说明剩下的链表不够k个,跳出循环
            if count:
                p.next = head
                break

            # 翻转操作
            while stack:
                p.next = stack.pop()
                p = p.next
            # 与剩下链表连接起来
            p.next = tmp
            head = tmp

        return dummy.next


class Solution26:
    """
    26. 删除有序数组中的重复项
    https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
    给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
    输入：nums = [1,1,2]
    输出：2, nums = [1,2,_]
    解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。

    输入：nums = [0,0,1,1,1,2,2,3,3,4]
    输出：5, nums = [0,1,2,3,4]
    解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。

    """

    @staticmethod
    def remove_duplicates(nums: List[int]) -> int:
        if not nums:
            return 0

        n = len(nums)
        fast = slow = 1
        while fast < n:
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        return slow


class Solution27:
    """
    27. 移除元素
    https://leetcode-cn.com/problems/remove-element/
    给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
    元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

    输入：nums = [3,2,2,3], val = 3
    输出：2, nums = [2,2]
    解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。

    输入：nums = [0,1,2,2,3,0,4,2], val = 2
    输出：5, nums = [0,1,4,0,3]
    解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。

    """

    # 双指针方法
    @staticmethod
    def remove_element(nums: List[int], val: int) -> int:
        n = len(nums)
        left = 0  # 左指针从0开始,指向下一个将要赋值的位置
        # 右指针从0开始,指向当前将要处理的元素
        for right in range(0, n):
            # 右指针指向的元素不等于val,是输出数组的元素
            # 将右指针指向的元素复制到左指针位置,然后将左右指针同时右移
            if nums[right] != val:
                nums[left] = nums[right]
                left += 1
                # 右指针指向的元素等于val,不在输出数组里,左指针不动,右指针右移一位
        return left  # left的值就是输出数组的长度

    # 双指针优化
    @staticmethod
    def remove_element_01(nums: List[int], val: int) -> int:
        left = 0  # 两个指针初始时分别位于数组的首尾
        right = len(nums)
        while left < right:
            # 左指针等于val,将右指针元素复制到左指针的位置,右指针左移一位
            if nums[left] == val:
                nums[left] = nums[right - 1]
                right -= 1
            else:  # 左指针不等于val,左指针右移一位,右指针不动
                left += 1
        return left  # left的值就是输出数组的长度


class Solution28:
    """
    28. 实现 strStr()
    https://leetcode-cn.com/problems/implement-strstr/
    给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

    输入：haystack = "hello", needle = "ll"
    输出：2

    输入：haystack = "aaaaa", needle = "bba"
    输出：-1

    """

    def str_str(self, haystack: str, needle: str) -> int:
        a = len(needle)
        b = len(haystack)
        if a == 0:
            return 0
        next = self.get_next(a, needle)
        p = -1
        for j in range(b):
            while p >= 0 and needle[p + 1] != haystack[j]:
                p = next[p]
            if needle[p + 1] == haystack[j]:
                p += 1
            if p == a - 1:
                return j - a + 1
        return -1

    @staticmethod
    def get_next(a, needle):
        next = ['' for _ in range(a)]
        k = -1
        next[0] = k
        for i in range(1, len(needle)):
            while k > -1 and needle[k + 1] != needle[i]:
                k = next[k]
            if needle[k + 1] == needle[i]:
                k += 1
            next[i] = k
        return next


class Solution29:
    """
    29. 两数相除
    https://leetcode-cn.com/problems/divide-two-integers/
    给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
    返回被除数 dividend 除以除数 divisor 得到的商。
    整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2

    输入: dividend = 10, divisor = 3
    输出: 3
    解释: 10/3 = truncate(3.33333..) = truncate(3) = 3

    输入: dividend = 7, divisor = -3
    输出: -2
    解释: 7/-3 = truncate(-2.33333..) = -2

    """

    @staticmethod
    def divide(dividend: int, divisor: int) -> int:
        INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1

        # 考虑被除数为最小值的情况
        if dividend == INT_MIN:
            if divisor == 1:
                return INT_MIN
            if divisor == -1:
                return INT_MAX

        # 考虑除数为最小值的情况
        if divisor == INT_MIN:
            return 1 if dividend == INT_MIN else 0
        # 考虑被除数为 0 的情况
        if dividend == 0:
            return 0

        # 一般情况，使用二分查找
        # 将所有的正数取相反数，这样就只需要考虑一种情况
        rev = False
        if dividend > 0:
            dividend = -dividend
            rev = not rev
        if divisor > 0:
            divisor = -divisor
            rev = not rev

        # 快速乘
        def quick_add(y: int, z: int, x: int) -> bool:
            # x 和 y 是负数，z 是正数
            # 需要判断 z * y >= x 是否成立
            result, add = 0, y
            while z > 0:
                if (z & 1) == 1:
                    # 需要保证 result + add >= x
                    if result < x - add:
                        return False
                    result += add
                if z != 1:
                    # 需要保证 add + add >= x
                    if add < x - add:
                        return False
                    add += add
                # 不能使用除法
                z >>= 1
            return True

        left, right, ans = 1, INT_MAX, 0
        while left <= right:
            # 注意溢出，并且不能使用除法
            mid = left + ((right - left) >> 1)
            check = quick_add(divisor, mid, dividend)
            if check:
                ans = mid
                # 注意溢出
                if mid == INT_MAX:
                    break
                left = mid + 1
            else:
                right = mid - 1

        return -ans if rev else ans


class Solution30:
    """
    30. 串联所有单词的子串
    https://leetcode-cn.com/problems/substring-with-concatenation-of-all-words/
    给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
    注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。

    输入：s = "barfoothefoobarman", words = ["foo","bar"]
    输出：[0,9]
    解释：
    从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
    输出的顺序不重要, [9,0] 也是有效答案。


    输入：s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
    输出：[6,9,12]
    """

    @staticmethod
    def find_substring(s: str, words: List[str]) -> List[int]:
        word_map = {}
        for word in words:
            word_map[word] = word_map.get(word, 0) + 1
        one_word_size = len(words[0])
        words_size = one_word_size * len(words)
        ans = []
        for i in range(0, len(s) - words_size + 1):
            temp_map = {}
            sub_s = s[i:i + words_size]
            flag = True
            for j in range(0, len(sub_s), one_word_size):
                check_word = sub_s[j:j + one_word_size]
                check_size = temp_map.get(check_word, 0) + 1
                count_size = word_map.get(check_word, 0)
                temp_map[check_word] = check_size
                if check_size > count_size or count_size == 0:
                    flag = False
                    break
            if flag:
                ans.append(i)
        return ans


class Solution31:
    """
    31. 下一个排列
    https://leetcode-cn.com/problems/next-permutation/
    整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。

    例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
    整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

    例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
    类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
    而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
    给你一个整数数组 nums ，找出 nums 的下一个排列。

    必须 原地 修改，只允许使用额外常数空间。

    输入：nums = [1,2,3]
    输出：[1,3,2]

    输入：nums = [3,2,1]
    输出：[1,2,3]
    """

    @staticmethod
    def next_permutation(nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


class Solution32:
    """
    32. 最长有效括号
    https://leetcode-cn.com/problems/longest-valid-parentheses/
    给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
    输入：s = ")()())"
    输出：4
    解释：最长有效括号子串是 "()()"

    """

    @staticmethod
    def longest_valid_parentheses(s: str) -> int:
        stack = []
        i = 0
        ans = 0
        stack.append(-1)
        while i < len(s):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    ans = max(ans, i - stack[-1])
            i += 1
        return ans


class Solution33:
    """
    33. 搜索旋转排序数组
    https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
    整数数组 nums 按升序排列，数组中的值 互不相同 。
    在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
    给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

    输入：nums = [4,5,6,7,0,1,2], target = 0
    输出：4
    输入：nums = [4,5,6,7,0,1,2], target = 3
    输出：-1
    """

    @staticmethod
    def search(nums: List[int], target: int) -> int:
        if not nums:
            return -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = int((left + right) / 2)
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[len(nums) - 1]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1


class Solution34:
    """
    34. 在排序数组中查找元素的第一个和最后一个位置
    https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
    如果数组中不存在目标值 target，返回 [-1, -1]。

    输入：nums = [5,7,7,8,8,10], target = 8
    输出：[3,4]

    输入：nums = [5,7,7,8,8,10], target = 6
    输出：[-1,-1]
    """

    @staticmethod
    def find(nums: List[int], target: int, first):
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if (first == 1 and nums[mid] >= target) or (first == 0 and nums[mid] > target):
                high = mid - 1
            else:
                low = mid + 1
        if first == 1 and nums[low] == target:
            return low
        elif first == 0 and nums[high] == target:
            return high
        else:
            return -1

    def search_range(self, nums: List[int], target: int) -> List[int]:

        if len(nums) == 0 or nums[0] > target or nums[-1] < target:
            return [-1, -1]

        lm = self.find(nums, target, 1)
        rm = self.find(nums, target, 0)

        return [lm, rm]


class Solution35:
    """
    35. 搜索插入位置
    https://leetcode-cn.com/problems/search-insert-position/
    给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
    请必须使用时间复杂度为 O(log n) 的算法。

    输入: nums = [1,3,5,6], target = 5
    输出: 2

    输入: nums = [1,3,5,6], target = 7
    输出: 4
    """

    @staticmethod
    def search_insert(nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        ans = len(nums)
        while left <= right:
            mid = left + (right - left) // 2
            if target <= nums[mid]:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans


class Solution36:
    """
    36. 有效的数独
    https://leetcode-cn.com/problems/valid-sudoku/
    请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。
    数字 1-9 在每一行只能出现一次。
    数字 1-9 在每一列只能出现一次。
    数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）

    """

    @staticmethod
    def is_valid_su_do_ku(board: List[List[str]]) -> bool:
        rows = [[0] * 9 for i in range(9)]
        columns = [[0] * 9 for i in range(9)]
        sub_boxes = [[[0] * 9 for i in range(3)] for j in range(3)]
        for i in range(9):
            for j in range(9):
                c = board[i][j]
                if c == ".":
                    continue
                index = int(c) - 1
                rows[i][index] += 1
                columns[j][index] += 1
                sub_boxes[i // 3][j // 3][index] += 1
                if rows[i][index] > 1 or columns[j][index] > 1 or sub_boxes[i // 3][j // 3][index] > 1:
                    return False
        return True


class Solution37:
    """
    37. 解数独
    https://leetcode-cn.com/problems/sudoku-solver/
    编写一个程序，通过填充空格来解决数独问题。

    数独的解法需 遵循如下规则：
    数字 1-9 在每一行只能出现一次。
    数字 1-9 在每一列只能出现一次。
    数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
    数独部分空格内已填入了数字，空白格用 '.' 表示。
    """

    @staticmethod
    def solve_su_do_ku(board: List[List[str]]) -> None:
        def dfs(pos: int):
            nonlocal valid
            if pos == len(spaces):
                valid = True
                return

            i, j = spaces[pos]
            for digit in range(9):
                if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
                    board[i][j] = str(digit + 1)
                    dfs(pos + 1)
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = False
                if valid:
                    return

        line = [[False] * 9 for _ in range(9)]
        column = [[False] * 9 for _ in range(9)]
        block = [[[False] * 9 for _a in range(3)] for _b in range(3)]
        valid = False
        spaces = list()

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    spaces.append((i, j))
                else:
                    digit = int(board[i][j]) - 1
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True

        dfs(0)


class Solution38:
    """
    38. 外观数列
    https://leetcode-cn.com/problems/count-and-say/

    给定一个正整数 n ，输出外观数列的第 n 项。
    「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。

    输入：n = 4
    输出："1211"
    解释：
    countAndSay(1) = "1"
    countAndSay(2) = 读 "1" = 一 个 1 = "11"
    countAndSay(3) = 读 "11" = 二 个 1 = "21"
    countAndSay(4) = 读 "21" = 一 个 2 + 一 个 1 = "12" + "11" = "1211"

    """

    @staticmethod
    def count_and_say(n: int) -> str:
        prev = "1"
        for i in range(n - 1):
            curr = ""
            pos = 0
            start = 0

            while pos < len(prev):
                while pos < len(prev) and prev[pos] == prev[start]:
                    pos += 1
                curr += str(pos - start) + prev[start]
                start = pos
            prev = curr

        return prev


class Solution39:
    """
    39. 组合总和
    https://leetcode-cn.com/problems/combination-sum/
    给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
    candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
    对于给定的输入，保证和为 target 的不同组合数少于 150 个。

    输入：candidates = [2,3,6,7], target = 7
    输出：[[2,2,3],[7]]
    解释：
    2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
    7 也是一个候选， 7 = 7 。
    仅有这两种组合。

    输入: candidates = [2,3,5], target = 8
    输出: [[2,2,2,2],[2,3,3],[3,5]]

    """

    @staticmethod
    def combination_sum(candidates: List[int], target: int) -> List[List[int]]:

        def dfs(candidates, combine, target, index):
            if index == len(candidates):
                return
            if target == 0:
                ans.append(combine)
                return
            # // 直接跳过
            dfs(candidates, combine, target, index + 1)
            # // 选择当前数
            if target - candidates[index] >= 0:
                dfs(candidates, combine + [candidates[index]], target - candidates[index], index)

        ans = []
        combine = []
        dfs(candidates, combine, target, 0)
        return ans


class Solution40:
    """
    40. 组合总和 II
    https://leetcode-cn.com/problems/combination-sum-ii/
    给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    candidates 中的每个数字在每个组合中只能使用 一次 。

    输入: candidates = [10,1,2,7,6,1,5], target = 8,
    输出:
    [
    [1,1,6],
    [1,2,5],
    [1,7],
    [2,6]
    ]

    输入: candidates = [2,5,2,1,2], target = 5,
    输出:
    [
    [1,2,2],
    [5]
    ]

    """

    @staticmethod
    def combination_sum2(candidates: List[int], target: int) -> List[List[int]]:

        def dfs(index: int, target: int):
            nonlocal combine
            if target == 0:
                ans.append(combine[:])
                return
            if index == len(freq) or target < freq[index][0]:
                return

            dfs(index + 1, target)
            most = min(target // freq[index][0], freq[index][1])
            for i in range(1, most + 1):
                combine.append(freq[index][0])
                dfs(index + 1, target - i * freq[index][0])
            combine = combine[:-most]

        freq = sorted(collections.Counter(candidates).items())
        ans = list()
        combine = list()
        dfs(0, target)
        return ans


class Solution41:
    """
    41. 缺失的第一个正数
    https://leetcode-cn.com/problems/first-missing-positive/
    给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
    请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。


    输入：nums = [1,2,0]
    输出：3

    输入：nums = [3,4,-1,1]
    输出：2

    """

    # 3 应该放在索引为 2 的地方
    # 4 应该放在索引为 3 的地方

    def first_missing_positive(self, nums: List[int]) -> int:
        size = len(nums)
        for i in range(size):
            # 先判断这个数字是不是索引，然后判断这个数字是不是放在了正确的地方
            while 1 <= nums[i] <= size and nums[i] != nums[nums[i] - 1]:
                self.swap(nums, i, nums[i] - 1)

        for i in range(size):
            if i + 1 != nums[i]:
                return i + 1

        return size + 1

    @staticmethod
    def swap(nums, index1, index2):
        nums[index1], nums[index2] = nums[index2], nums[index1]


class Solution42:
    """
    42. 接雨水
    https://leetcode-cn.com/problems/trapping-rain-water/
    给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

    输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
    输出：6
    解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

    输入：height = [4,2,0,3,2,5]
    输出：9
    """

    @staticmethod
    def trap(height: List[int]) -> int:
        ans = 0
        stack = list()
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                curr_width = i - left - 1
                curr_height = min(height[left], height[i]) - height[top]
                ans += curr_width * curr_height
            stack.append(i)

        return ans


class Solution43:
    """
    43. 字符串相乘
    https://leetcode-cn.com/problems/multiply-strings/
    给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
    注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。
    输入: num1 = "2", num2 = "3"
    输出: "6"

    输入: num1 = "123", num2 = "456"
    输出: "56088"
    """

    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"

        ans = "0"
        m, n = len(num1), len(num2)
        for i in range(n - 1, -1, -1):
            add = 0
            y = int(num2[i])
            curr = ["0"] * (n - i - 1)
            for j in range(m - 1, -1, -1):
                product = int(num1[j]) * y + add
                curr.append(str(product % 10))
                add = product // 10
            if add > 0:
                curr.append(str(add))
            curr = "".join(curr[::-1])
            ans = self.add_strings(ans, curr)

        return ans

    @staticmethod
    def add_strings(num1: str, num2: str) -> str:
        i, j = len(num1) - 1, len(num2) - 1
        add = 0
        ans = list()
        while i >= 0 or j >= 0 or add != 0:
            x = int(num1[i]) if i >= 0 else 0
            y = int(num2[j]) if j >= 0 else 0
            result = x + y + add
            ans.append(str(result % 10))
            add = result // 10
            i -= 1
            j -= 1
        return "".join(ans[::-1])


class Solution44:
    """
    44. 通配符匹配
    https://leetcode-cn.com/problems/wildcard-matching/solution/tong-pei-fu-pi-pei-by-leetcode-solution/
    给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
    '?' 可以匹配任何单个字符。
    '*' 可以匹配任意字符串（包括空字符串）。
    两个字符串完全匹配才算匹配成功。

    说明:

    s 可能为空，且只包含从 a-z 的小写字母。
    p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。

    输入:
    s = "aa"
    p = "a"
    输出: false
    解释: "a" 无法匹配 "aa" 整个字符串。

    输入:
    s = "aa"
    p = "*"
    输出: true
    解释: '*' 可以匹配任意字符串。

    输入:
    s = "cb"
    p = "?a"
    输出: false
    解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。

    输入:
    s = "adceb"
    p = "*a*b"
    输出: true
    解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".

    """

    @staticmethod
    def is_match(s: str, p: str) -> bool:
        m, n = len(s), len(p)

        dp = [[False] * (n + 1) for _ in range(m + 1)]  # dp[i][0]=False，即空模式无法匹配非空字符串；
        dp[0][0] = True  # dp[0][0]=True，即当字符串 ss 和模式 pp 均为空时，匹配成功
        for i in range(1, n + 1):  # dp[0][j] 需要分情况讨论：因为星号才能匹配空字符串，所以只有当模式p的前j个字符均为星号时，dp[0][j] 才为真。
            if p[i - 1] == '*':
                dp[0][i] = True
            else:
                break

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] | dp[i - 1][j]  # 不使用这个星号 dp[i][j - 1]， 使用这个星号 dp[i - 1][j]
                elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:  # 问号和2个字母完全匹配的情况
                    dp[i][j] = dp[i - 1][j - 1]

        return dp[m][n]


class Solution45:
    """
    45. 跳跃游戏 II
    https://leetcode-cn.com/problems/jump-game-ii/
    给你一个非负整数数组 nums ，你最初位于数组的第一个位置。
    数组中的每个元素代表你在该位置可以跳跃的最大长度。
    你的目标是使用最少的跳跃次数到达数组的最后一个位置。
    假设你总是可以到达数组的最后一个位置。

    nums = [2,3,1,1,4]
    跳到最后一个位置的最小跳跃数是 2。
    从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

    """

    @staticmethod
    def jump(nums: List[int]) -> int:
        n = len(nums)
        max_pos, end, step = 0, 0, 0
        for i in range(n - 1):
            if max_pos >= i:
                max_pos = max(max_pos, i + nums[i])  # 维护当前能够到达的最大下标位置，记为边界。我们从左到右遍历数组，到达边界时，更新边界并将跳跃次数增加 1
                if i == end:
                    end = max_pos
                    step += 1
        return step


class Solution46:
    """
    46. 全排列
    https://leetcode-cn.com/problems/permutations/
    给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

    输入：nums = [1,2,3]
    输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

    """

    @staticmethod
    def permute(nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def backtrack(first=0):
            # 所有数都填完了
            if first == len(nums):
                res.append(nums[:])
            for i in range(first, len(nums)):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]

        res = []
        backtrack()
        return res


class Solution47:
    """
    47. 全排列 II
    https://leetcode-cn.com/problems/permutations-ii/
    给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

    输入：nums = [1,1,2]
    输出：
    [[1,1,2],
     [1,2,1],
     [2,1,1]]
    """

    def __init__(self):
        self.res = []

    def permute_unique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        check = [0 for _ in range(len(nums))]

        self.backtrack([], nums, check)
        return self.res

    def backtrack(self, sol, nums, check):
        if len(sol) == len(nums):
            self.res.append(sol)
            return

        for i in range(len(nums)):
            if check[i] == 1:
                continue
            if i > 0 and nums[i] == nums[i - 1] and check[i - 1] == 0:
                continue
            check[i] = 1
            self.backtrack(sol + [nums[i]], nums, check)
            check[i] = 0


class Solution48:
    """
    48. 旋转图像
    https://leetcode-cn.com/problems/rotate-image/
    给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
    你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
    输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
    输出：[[7,4,1],[8,5,2],[9,6,3]]
    """

    @staticmethod
    def rotate(matrix: List[List[int]]) -> None:
        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        # 主对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


class Solution49:
    """
    49. 字母异位词分组
    https://leetcode-cn.com/problems/group-anagrams/
    给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
    字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

    输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

    """

    @staticmethod
    def group_anagrams(str_list: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)

        for st in str_list:
            key = "".join(sorted(st))
            mp[key].append(st)

        return list(mp.values())


class Solution50:
    """
    50. Pow(x, n)
    https://leetcode-cn.com/problems/powx-n/
    实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn ）。
    输入：x = 2.00000, n = 10
    输出：1024.00000

    输入：x = 2.10000, n = 3
    输出：9.26100
    """

    @staticmethod
    def myPow(x: float, n: int) -> float:
        def quickMul(N):
            if N == 0:
                return 1.0
            y = quickMul(N // 2)
            return y * y if N % 2 == 0 else y * y * x

        return quickMul(n) if n >= 0 else 1.0 / quickMul(-n)

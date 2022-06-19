import sys
import heapq
import itertools
import collections
import functools
from typing import List, Optional
from sortedcontainers import SortedList

class DivideAndConquer:
    """分治算法案列"""

    def __init__(self):
        self.num = 0

    def count_reverse_size(self, a_list, low, high):
        """计算数组的逆序度"""
        if low >= high:
            return
        mid = (low + high) / 2
        self.count_reverse_size(a_list, low, mid)
        self.count_reverse_size(a_list, mid + 1, high)

        self.merge_reverse_size(a_list, low, mid, high)

    def merge_reverse_size(self, a_list, low, mid, high):
        """计算数组的逆序度指归并"""
        i = low
        j = mid + 1
        k = 0
        temp = [None] * (high - low + 1)  # 构建一个临时的数组
        while i <= mid and j <= high:  # 用数组的前半部分和后半部分依次比较
            if a_list[i] <= a_list[j]:  # 如果前半部分的元素小于后半部分的
                temp[k] = a_list[i]  # 将前部分的元素放到临时数组中，后半部分索引不变，前半部分所以加 1
                k += 1
                i += 1
            else:  # 如果前半部分的元素大于后半部分的
                self.num += mid - i + 1  # 前半部分剩下的元素都比 a_list[j] 大，这些元素的个数就是逆序度
                temp[k] = a_list[j]
                k += 1
                j += 1
        while i <= mid:  # 处理前部分剩下的元素
            temp[k] = a_list[i]
            k += 1
            i += 1
        while j <= high:  # 处理后半部分剩下的元素
            temp[k] = a_list[j]
            k += 1
            j += 1
        for i in range(0, high - low + 1):
            a_list[low + i] = temp[i]

    @staticmethod
    def count_reverse_size_v2(nums) -> int:
        temp = nums[:]
        temp.sort()
        result = 0
        for i, v in enumerate(nums):
            a = temp.index(v)
            result += a
            temp.pop(a)
        return result

    # from sortedcontainers import SortedList
    @staticmethod
    def count_reverse_size_v3(nums) -> int:  # 维护一个有序数组 sl，从右往左依次往里添加 nums 中的元素，
                                            # 每次添加 nums[i] 前基于「二分搜索」判断出当前 sl 中比 nums[i]
                                            # 小的元素个数（即 nums[i] 右侧比 nums[i] 还要小的元素个数），并累计入答案即可。
        n = len(nums)
        sl = SortedList()

        ans = 0
        for i in range(n - 1, -1, -1):  # 反向遍历
            cnt = sl.bisect_left(nums[i])  # 找到右边比当前值小的元素个数
            ans += cnt  # 记入答案
            sl.add(nums[i])  # 将当前值加入有序数组中

        return ans



class Backpack:
    def __init__(self):
        self.max_weight = -1  # tracking the max weight
        self.max_value = -1  # tracking the max value

    def backpack(self, i, cw, items, w):
        """
        只满足背包重量最大
        0-1 背包 问题
        :param i: the ith item, integer
        :param cw:  current weight, integer
        :param items:  python list of item weights
        :param w: upper limit weight the backpack can load
        :return:
        """

        if cw == w or i == len(items):  # base case
            if cw > self.max_weight:
                self.max_weight = cw
            return

        self.backpack(i + 1, cw, items, w)  # 递归调用表示不选择当前物品，直接考虑下一个（第 i+1 个），故 cw 不更新
        if cw + items[i] <= w:
            self.backpack(i + 1, cw + items[i], items, w)  # 递归调用表示选择了当前物品，故考虑下一个时，cw 通过入参更新为 cw + items[i]

    def backpack2(self, i, cw, items, w):
        """
        只满足背包重量最大,用一个二维数组记录之前已经计算的值，避免重复计算
        0-1 背包 问题
        :param i: the ith item, integer
        :param cw:  current weight, integer
        :param items:  python list of item weights
        :param w: upper limit weight the backpack can load
        :return:
        """
        mem = [[False for _ in range(w + 2)] for _ in range(len(items) + 1)]
        if cw == w or i == len(items):  # base case
            if cw > self.max_weight:
                self.max_weight = cw
            return
        if mem[i][cw]:
            return
        mem[i][cw] = True
        self.backpack2(i + 1, cw, items, w)  # 递归调用表示不选择当前物品，直接考虑下一个（第 i+1 个），故 cw 不更新
        if cw + items[i] <= w:
            self.backpack2(i + 1, cw + items[i], items, w)  # 递归调用表示选择了当前物品，故考虑下一个时，cw 通过入参更新为 cw + items[i]

    def backpack3(self, i, w, cw, cv, items, values):
        """
        在满足背包最大重量限制的前提下，求背包中可装入物品的总价值最大是多少？
        0-1 背包 问题
        :param i: the ith item, integer
        :param cw:  current weight, integer
        :param items:  python list of item weights
        :param values:  python list of item values
        :param w: upper limit weight the backpack can load
        :param cv:  current value, integer
        :return:
        """
        if cw == w or i == len(items):  # cw == w 表示装满了， i==n 表示物品都考察完了
            if cv > self.max_value:
                self.max_value = cv
            return

        self.backpack3(i + 1, w, cw, cv, items, values)  # 递归调用表示不选择当前物品，直接考虑下一个（第 i+1 个），故 cw 不更新
        if cw + items[i] <= w:
            # 递归调用表示选择了当前物品，故考虑下一个时，cw 通过入参更新为 cw + items[i]
            self.backpack3(i + 1, w, cw + items[i], cv + values[i], items, values)


class Dijkstra:
    """戴克斯特拉算法"""
    def __init__(self):
        self.graph = {}  # 记录地图指向关系，和相应的权重
        self.costs = {}  # 记录从起点到每个点的距离
        self.parents = {}  # 记录每个点的父节点
        self.processed = []  # 记录已经处理里的节点

    def add_edge(self, s, t, w):  # s 先于 t, 边 s -> t
        if not self.graph.get(s):
            self.graph[s] = {}
        if t and w:
            self.graph[s][t] = w

    def dijkstra(self):
        node = self.find_lowest_cost_node()  # 在未处理的节点中找出开销最小的节点
        while node is not None:  # while循环在所有节点都被处理过后结束
            cost = self.costs[node]
            neighbors = self.graph[node]
            for n in neighbors.keys():
                new_cost = cost + neighbors[n]
                if self.costs[n] > new_cost:  # 如果经当前节点前往该邻居更近
                    self.costs[n] = new_cost  # 更新该邻居节点的开销
                    self.parents[n] = node  # 该邻居节点的父节点设置为当前节点
            self.processed.append(node)  # 将当前节点标记为处理过
            node = self.find_lowest_cost_node()  # 找出接下来要处理的节点，并循环
        print(self.costs)
        print(self.parents)

    def find_lowest_cost_node(self):
        lowest_cost = float("inf")
        lowest_cost_node = None
        for node in self.costs:  # 遍历所有的节点
            cost = self.costs[node]
            if cost < lowest_cost and node not in self.processed:
                lowest_cost = cost  # 就将其视为开销最低的节点
                lowest_cost_node = node
        return lowest_cost_node


class Solution1:
    """
    5. 最长回文子串
    https://leetcode-cn.com/problems/longest-palindromic-substring/

    输入：s = "babad"
    输出："bab"
    解释："aba" 同样是符合题意的答案。

    输入：s = "cbbd"
    输出："bb"

    """

    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]

    def longest_palindrome(self, s: str) -> str:
        res = ''
        for i in range(len(s)):
            # 先判定奇数的，从i开始左右对比
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # 再判定偶数的，从i和i+1开始对比
            tmp = self.helper(s, i, i + 1)
            if len(tmp) > len(res):
                res = tmp
        return res


class Solution2:
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


class Solution3:
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

class Solution4:
    """
    22. 括号生成
    https://leetcode-cn.com/problems/generate-parentheses/
    数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

    输入：n = 3
    输出：["((()))","(()())","(())()","()(())","()()()"]

    """

    @staticmethod
    def generate_parenthesis(n: int) -> List[str]:
        ans = []

        def backtrack(s, left, right):
            if len(s) == 2 * n:
                ans.append(''.join(s))
                return

            if left < n:  # 如果左括号数量不大于 n，我们可以放一个左括号
                s.append('(')
                backtrack(s, left + 1, right)
                s.pop()  # 回朔

            if right < left:  # 如果右括号数量小于左括号的数量，我们可以放一个右括号
                s.append(')')
                backtrack(s, left, right + 1)
                s.pop()  # 回朔

        backtrack([], 0, 0)
        return ans


class Solution5:
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
        if not dividend:
            return 0
        if dividend == -2 ** 31 and divisor == -1:
            return 2 ** 31 - 1
        flag = 0 if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0) else 1

        dividend, divisor = abs(dividend), abs(divisor)
        result = 0
        for n in range(31, -1, -1):
            if (divisor << n) <= dividend:
                result += 1 << n
                dividend -= divisor << n
                continue
        return result if flag else -result


class Solution6:
    """
    32. 最长有效括号
    https://leetcode-cn.com/problems/longest-valid-parentheses/
    给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
    输入：s = ")()())"
    输出：4
    解释：最长有效括号子串是 "()()"

    始终保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」
    当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
    """

    @staticmethod
    def longest_valid_parentheses(s: str) -> int:
        stack = []
        i = 0
        ans = 0
        stack.append(-1)
        while i < len(s):
            if s[i] == "(":  # 对于遇到的每个(，我们将它的下标放入栈中
                stack.append(i)
            else:
                stack.pop()  # 对于遇到的每个)，我们先弹出栈顶元素表示匹配了当前右括号
                if len(stack) == 0:  # 如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
                    stack.append(i)
                else:
                    ans = max(ans, i - stack[-1])  # 如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
            i += 1
        return ans


class Solution7:
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
        block = [[[False] * 9 for _ in range(3)] for _ in range(3)]
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

class Solution8:
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

        def dfs(combine, target, index):
            if index == len(candidates):
                return
            if target == 0:
                ans.append(combine)
                return

            dfs(combine, target, index + 1)  # 直接跳过

            if target - candidates[index] >= 0:
                dfs(combine + [candidates[index]], target - candidates[index], index)  # 选择当前数

        ans = []
        combine = []
        dfs(combine, target, 0)
        return ans

class Solution9:
    """
    40. 组合总和 II
    https://leetcode-cn.com/problems/combination-sum-ii/
    给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    candidates 中的每个数字在每个组合中只能使用 一次 。
    """
    @staticmethod
    def combination_sum2(candidates: List[int], target: int) -> List[List[int]]:
        def dfs(combine, target, index):
            if target == 0:
                res.append(combine[:])
                return

            for i in range(index, len(candidates)):
                if candidates[i] > target:
                    break

                if i > index and candidates[i - 1] == candidates[i]:
                    continue

                combine.append(candidates[i])
                dfs(combine, target - candidates[i], i + 1)
                combine.pop()

        if len(candidates) == 0:
            return []

        candidates.sort()
        res = []
        combine = []
        dfs(combine, target, 0)
        return res

class Solution10:
    """
    组合
    https://leetcode-cn.com/problems/combinations/
    给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
    """

    @staticmethod
    def combine(n: int, k: int) -> List[List[int]]:
        result = []  # 存放结果
        path = []  # 存放路径

        def backtracking(n, k, startIndex):
            if k == len(path):
                result.append(path[:])
                return
            for i in range(startIndex, n + 1):
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()  ## 回溯，撤销处理结果

        backtracking(n, k, 1)
        return result

    @staticmethod
    def combine_v2(n: int, k: int) -> List[List[int]]:
        aa = [x for x in range(1, n + 1)]
        bb = itertools.combinations(aa, k)
        result = []
        for x in bb:
            result.append(x)
        return result



class Solution11:
    """
    子集
    https://leetcode-cn.com/problems/subsets/
    给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
    输入：nums = [1,2,3]
    输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    """

    @staticmethod
    def subsets_v1(nums: List[int]) -> List[List[int]]:
        res = []
        for i in range(len(nums) + 1):
            for tmp in itertools.combinations(nums, i):
                res.append(tmp)
        return res

    @staticmethod
    def subsets_v2(nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
        return res

class Solution12(object):
    """
    子集 II
    https://leetcode-cn.com/problems/subsets-ii/
    给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
    解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
    """

    def subsets_with_dup(self, nums):
        res = []
        nums.sort()
        self.dfs(nums, 0, res, [])
        return res

    def dfs(self, nums, index, res, path):
        if path not in res:
            res.append(path)
        for i in range(index, len(nums)):
            if i > index and nums[i] == nums[i - 1]:
                continue
            self.dfs(nums, i + 1, res, path + [nums[i]])

class Solution13:
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
    def trap(height: List[int]) -> int:  # 双指针法
        ans = 0
        left, right = 0, len(height) - 1
        left_max = right_max = 0

        while left < right:
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])
            if height[left] < height[right]:
                ans += left_max - height[left]
                left += 1
            else:
                ans += right_max - height[right]
                right -= 1

        return ans

class Solution14:
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

class Solution15:
    """
    47. 全排列 II
    https://leetcode-cn.com/problems/permutations-ii/
    给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
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



class Solution16:
    """
    不同路径II
    https://leetcode-cn.com/problems/unique-paths-ii/
    个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

    机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
    现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
    网格中的障碍物和空位置分别用 1 和 0 来表示。
    """

    @staticmethod
    def unique_paths_with_obstacles(obstacle_grid: List[List[int]]) -> int:
        m, n = len(obstacle_grid), len(obstacle_grid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(n):
            if obstacle_grid[0][i] == 1:
                break
            dp[0][i] = 1

        for j in range(m):
            if obstacle_grid[j][0] == 1:
                break
            dp[j][0] = 1

        for i in range(1, m):
            for j in range(1, n):
                if obstacle_grid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
        return dp[m - 1][n - 1]


class Solution17:
    """
    编辑距离
    https://leetcode-cn.com/problems/edit-distance/
    给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数 。

    你可以对一个单词进行如下三种操作：

    插入一个字符
    删除一个字符
    替换一个字符
    """

    @staticmethod
    def min_distance(word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)

        # 有一个字符串为空串
        if n * m == 0:
            return n + m

        # DP 数组
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # 边界状态初始化
        for i in range(n + 1):
            dp[i][0] = i

        for j in range(m + 1):
            dp[0][j] = j

        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = dp[i - 1][j] + 1
                down = dp[i][j - 1] + 1
                left_down = dp[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                dp[i][j] = min(left, down, left_down)

        return dp[n][m]

class Solution18:
    """
    最小覆盖子串
    https://leetcode-cn.com/problems/minimum-window-substring/
    给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

    用i,j表示滑动窗口的左边界和右边界，通过改变i,j来扩展和收缩滑动窗口，可以想象成一个窗口在字符串上游走，当这个窗口包含的元素满足条件，
    即包含字符串T的所有元素，记录下这个滑动窗口的长度j-i+1，这些长度中的最小值就是要求的结果。
    一，不断增加j使滑动窗口增大，直到窗口包含了T的所有元素
    二，不断增加i使滑动窗口缩小，因为是要求最小字串，所以将不必要的元素排除在外，使长度减小，直到碰到一个必须包含的元素，这个时候不能再扔了，再扔就不满足条件了，记录此时滑动窗口的长度，并保存最小值
    三，让i再增加一个位置，这个时候滑动窗口肯定不满足条件了，那么继续从步骤一开始执行，寻找新的满足条件的滑动窗口，如此反复，直到j超出了字符串S范围。
    """

    @staticmethod
    def min_window(s: str, t: str) -> str:
        need = collections.defaultdict(int)  # 用一个字典need来表示当前滑动窗口中需要的各元素的数量，
        for c in t:
            need[c] += 1   # 一开始滑动窗口为空，用T中各元素来初始化这个need
        need_cnt = len(t)  # 维护一个额外的变量need_cnt来记录所需元素的总数量，
        i = 0
        res = (0, float('inf'))
        for j, c in enumerate(s):
            if need[c] > 0:  # 当我们碰到一个所需元素c，不仅need[c]的数量减少1，同时need_cnt也要减少1，这样我们通过need_cnt就可以知道是否满足条件
                need_cnt -= 1
            need[c] -= 1    # 如果 c 不在 t 中，那么 need[c] 会小于 0
            if need_cnt == 0:  # 步骤一：滑动窗口包含了所有T元素
                while True:  # 步骤二：增加i，排除多余元素
                    c = s[i]
                    if need[c] == 0: # 碰到一个必须包含的元素
                        break
                    need[c] += 1 # 扔掉了一个，所有这里要 need[c] 要加1
                    i += 1
                if j - i < res[1] - res[0]:  # 记录结果
                    res = (i, j)
                need[s[i]] += 1  # 步骤三：i增加一个位置，寻找新的满足条件滑动窗口
                need_cnt += 1
                i += 1
        return '' if res[1] > len(s) else s[res[0]:res[1] + 1]  # 如果res始终没被更新过，代表无满足条件的结果


class Solution19:
    """
    单词搜索
    https://leetcode-cn.com/problems/word-search/
    给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
    单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用
    """

    @staticmethod
    def exist(board, word):
        row, col = len(board), len(board[0])

        def dfs(x, y, index):
            if board[x][y] != word[index]:
                return False
            if index == len(word) - 1:
                return True
            board[x][y] = '#'
            for choice in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                nx, ny = x + choice[0], y + choice[1]
                if 0 <= nx < row and 0 <= ny < col and dfs(nx, ny, index + 1):
                    return True
            board[x][y] = word[index]

        for i in range(row):
            for j in range(col):
                if dfs(i, j, 0):
                    return True
        return False


class Solution20:
    """
    柱状图中最大的矩形
    https://leetcode-cn.com/problems/largest-rectangle-in-histogram/
    给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
    求在该柱状图中，能够勾勒出来的矩形的最大面积。
    首先我们枚举某一根柱子 i 作为高 h=heights[i]；
    随后我们需要进行向左右两边扩展，使得扩展到的柱子的高度均不小于 h。换句话说，我们需要找到左右两侧最近的高度小于 h 的柱子，这样这两根柱子之间（不包括其本身）的所有柱子高度均不小于 h，并且就是 i 能够扩展到的最远范围。
    输入：[6,7,5,2,4,5,9,3]
    左侧的柱子编号分别为 [−1,0,−1,−1,3,4,5,3]
    右侧的柱子编号分别为 [2,2,3,8,7,7,7,8]
    """

    @staticmethod
    def largest_rectangle_area(heights: List[int]) -> int:
        n = len(heights)
        left, right = [0] * n, [0] * n

        mono_stack = list()
        for i in range(n):  # 从左往右对数组进行遍历，借助单调栈求出了每根柱子的左边界(左侧且最近的小于其高度的柱子)
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)

        mono_stack = list()
        for i in range(n - 1, -1, -1):  # 从右往左对数组进行遍历，借助单调栈求出了每根柱子的右边界((右侧且最近的小于其高度的柱子))
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            right[i] = mono_stack[-1] if mono_stack else n
            mono_stack.append(i)

        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
        return ans



class Solution21:
    """
    最大矩形
    https://leetcode-cn.com/problems/maximal-rectangle/
    给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
    """

    @staticmethod
    def largest_rectangle_area(heights: List[int]) -> int:
        n = len(heights)
        left, right = [0] * n, [n] * n

        mono_stack = list()
        for i in range(n):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                right[mono_stack[-1]] = i  # 此时 mono_stack[-1] 位置的右边界是 i, 因为如果还有其他高度小于等于heights[mono_stack[-1]]的，已经出栈了
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)

        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
        return ans

    def max_rectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        # 记录当前位置上方连续“1”的个数
        pre = [0] * (n + 1)
        res = 0
        for i in range(m):
            for j in range(n):
                # 前缀和
                pre[j] = pre[j] + 1 if matrix[i][j] == "1" else 0
            ans = self.largest_rectangle_area(pre)
            res = max(res, ans)
        return res


class Solution22:
    """
    扰乱字符串
    https://leetcode-cn.com/problems/scramble-string/
    使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：
    如果字符串的长度为 1 ，算法停止
    如果字符串的长度 > 1 ，执行下述步骤：
        在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。
        随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。
        在 x 和 y 这两个子字符串上继续从步骤 1 开始递归执行此算法。
    给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 。
    """

    @functools.lru_cache(None)
    def is_scramble(self, s1: str, s2: str) -> bool:
        n = len(s1)
        if n == 0:
            return True
        if n == 1:
            return s1 == s2
        if sorted(s1) != sorted(s2):
            return False
        for i in range(1, n):
            if self.is_scramble(s1[:i], s2[:i]) and self.is_scramble(s1[i:], s2[i:]):
                return True
            elif self.is_scramble(s1[i:], s2[:-i]) and self.is_scramble(s1[:i], s2[-i:]):
                return True
        return False



class Solution23:
    """
    解码方法
    动态规划
    https://leetcode-cn.com/problems/decode-ways/
    一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
    'A' -> "1"
    'B' -> "2"
    'Z' -> "26"
    要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
    "AAJF" ，将消息分组为 (1 1 10 6)
    "KJF" ，将消息分组为 (11 10 6)
    注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
    给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
    题目数据保证答案肯定是一个 32 位 的整数。
    状态转移方程为： f[i] = f[i - 1] + f[i - 2] 但需要满足一定条件
    """

    @staticmethod
    def num_decode(s: str) -> int:
        n = len(s)
        f = [1] + [0] * n
        for i in range(1, n + 1):
            if s[i - 1] != '0':  # 使用了1 个字符，即 s[i] 进行解码，只要 s[i-1] ！=0 它就可以被解码成 A∼I 中的某个字母
                f[i] = f[i - 1]
            if i > 1 and s[i - 2] != '0' and int(s[i - 2:i]) <= 26:
                f[i] += f[i - 2] # 使用了2个字符
        return f[n]


class Solution24:
    """
    复原 IP 地址
    https://leetcode-cn.com/problems/restore-ip-addresses/
    有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
    例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
    给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。
    """

    @staticmethod
    def restore_ip_addresses(s: str) -> list:
        """
        一共四位，每一位都可以取 0 - 255之间这些数字，也就是，每一位都可以取 1 - 3位，也就是每一位都有三种取法。
        抽象出来就是四层，三叉树。
        从中去掉不符合规则的就可以了。
        """
        if len(s) < 4:
            return []

        rst = []

        def get_ip(ip_list, idx):
            # 4. 长度大于 4 剪枝
            if len(ip_list) == 4:
                # 5. 正好取完，才对
                if idx == len(s):
                    rst.append('.'.join(ip_list))
                return

            for i in range(1, 4):
                # 3. 这个可加可不加，是因为 假如 'ab' 切片长度大于 2，都是 取ab，那么会取到两个相同结果，但是5会限制，只有idx == len(s) 才会取
                if idx + i > len(s):
                    continue
                sub = s[idx: idx + i]
                # 1. 包含前导0， 剪枝
                if len(sub) > 1 and sub[0] == '0':
                    continue
                # 2. 当取得这一位大于255，直接剪枝
                if int(sub) > 255:
                    continue
                ip_list.append(sub)
                get_ip(ip_list, idx + i)
                ip_list.pop()

        get_ip([], 0)
        return rst


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution25:
    """
    不同的二叉搜索树 II
    https://leetcode-cn.com/problems/unique-binary-search-trees-ii/
    给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。
    """

    @staticmethod
    def generate_trees(n: int) -> List[TreeNode]:
        if n == 0:
            return []

        @functools.lru_cache()
        def helper(start, end):
            res = []
            if start > end:
                res.append(None)
            # 这里找根节点 取遍 [start, end] 中的节点
            for val in range(start, end + 1):
                # 这里寻找 左子树, helper(start, val-1) 返回 根节点 val 左半部分[start, val-1] 所能构成的所有左子树
                for left in helper(start, val - 1):
                    # 这里寻找 右子树, helper(val+1, end) 返回 根节点 val 右半部分[val+1, end] 所能构成的所有右子树
                    for right in helper(val + 1, end):
                        root = TreeNode(val)
                        root.left = left
                        root.right = right
                        # 保存 一种 { 左子树 | 根节点 | 右子树} 组合
                        res.append(root)
            return res

        return helper(1, n)


class Solution26:
    """
    不同的二叉搜索树
    https://leetcode-cn.com/problems/unique-binary-search-trees/
    给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

    设 f(n) 为 n 个节点的二叉树组合数。假设 n=5。当左子树共有 1 个节点的时候，一共有 f(1)*f(5-1-1) 个二叉树的组合；
    左子树共有 2 个节点的时候，一共有 f(2)*f(5-1-2) 个二叉树的组合；以此类推，当左子树共有 i 个节点的时候，一共有 f(i)*f(n-1-i) 个组合，
    即左子树组合数x右子树组合数。因此，想求所有组合，就把从 0 到 n-1 的左子树和右子树的组合都求出来即可。
    """

    @staticmethod
    def num_trees(n: int) -> int:
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(0, i):
                dp[i] += dp[j] * dp[i - j - 1]  # dp[5] = dp[0]*dp[4] + dp[1]*dp[3] + dp[2]*dp[2] + dp[3]*dp[1] + dp[4]*dp[0]
        return dp[n]


class Solution27:
    """
    交错字符串
    https://leetcode-cn.com/problems/interleaving-string/
    给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
    两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：
    输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
    输出：true
    """

    @staticmethod
    def is_interleave(s1: str, s2: str, s3: str) -> bool:
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)
        if len1 + len2 != len3:
            return False
        dp = [[False] * (len2 + 1) for _ in range(len1 + 1)]
        dp[0][0] = True

        for i in range(1, len1 + 1):
            dp[i][0] = (dp[i - 1][0] and s1[i - 1] == s3[i - 1]) # 表示 s1 的前 i 位是否能构成 s3 的前 i 位。因此需要满足的条件为，
                                                                 # 前 i−1 位可以构成 s3 的前 i−1 位且 s1 的第 i 位（s1[i−1]）等于 s3 的第 i 位（s3[i−1]）


        for j in range(1, len2 + 1):
            dp[0][j] = (dp[0][j - 1] and s2[j - 1] == s3[j - 1])  # 表示 s2 的前 i 位是否能构成 s3 的前 i 位。因此需要满足的条件为
                                                                  # 前 i−1 位可以构成 s3 的前 i−1 位且 s2 的第 i 位（s1[i−1]）等于 s3 的第 i 位（s3[i−1]）

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):  # s1 的前 i 个字符和 s2 的前 j−1 个字符能否构成 s3 的前 i+j−1 位，且 s2 的第 j 位（s2[j−1]）是否等于 s3 的第 i+j 位（s3[i+j−1]）。
                dp[i][j] = (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]) or (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1])
        return dp[-1][-1]


class Solution28:
    """
    验证二叉搜索树
    https://leetcode-cn.com/problems/validate-binary-search-tree/
    给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
    有效 二叉搜索树定义如下：
        节点的所有左子树只包含 小于 当前节点的数。
        节点的所有右子树只包含 大于 当前节点的数。
        所有左子树和右子树自身必须也是二叉搜索树。
    """

    @staticmethod
    def is_valid_bst(root: TreeNode) -> bool:
        def check_is_valid_bst(node, min_val=float('-inf'), max_val=float('inf')) -> bool:
            if not node:
                return True
            if node.val <= min_val or node.val >= max_val:
                return False
            if not check_is_valid_bst(node.left, min_val, node.val):
                return False
            if not check_is_valid_bst(node.right, node.val, max_val):
                return False
            return True

        return check_is_valid_bst(root)

    @staticmethod
    def is_valid_bst_v2(root: TreeNode) -> bool:
        stack, pre_order = [], float('-inf')

        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            # 如果中序遍历得到的节点的值小于等于前一个 in_order，说明不是二叉搜索树
            if root.val <= pre_order:
                return False
            pre_order = root.val
            root = root.right
        return True

    @staticmethod
    def is_valid_bst_v3(root: TreeNode) -> bool:
        res = []

        def dfs(root):
            if not root:
                return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)

        dfs(root)
        for i in range(1, len(res)):
            if res[i - 1] >= res[i]:
                return False
        return True


class Solution29(object):
    """
    恢复二叉搜索树
    https://leetcode-cn.com/problems/recover-binary-search-tree/
    给你二叉搜索树的根节点 root ，该树中的 恰好 两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树 。
    """

    @staticmethod
    def recover_tree(root):
        nodes = []

        # 中序遍历二叉树，并将遍历的结果保存到list中
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            nodes.append(root)
            dfs(root.right)

        dfs(root)
        x = None
        y = None
        pre = nodes[0]
        # 扫面遍历的结果，找出可能存在错误交换的节点x和y
        for i in range(1, len(nodes)):
            if pre.val > nodes[i].val:
                y = nodes[i]
                if not x:
                    x = pre
            pre = nodes[i]
        # 如果x和y不为空，则交换这两个节点值，恢复二叉搜索树
        if x and y:
            x.val, y.val = y.val, x.val


class Solution30(object):
    """
    二叉树的层序遍历
    https://leetcode-cn.com/problems/binary-tree-level-order-traversal/
    给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
    """

    @staticmethod
    def level_order(root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            # 获取当前队列的长度，这个长度相当于 当前这一层的节点个数
            size = len(queue)
            tmp = []
            # 将队列中的元素都拿出来(也就是获取这一层的节点)，放到临时list中
            # 如果节点的左/右子树不为空，也放入队列中
            for _ in range(size):
                r = queue.pop(0)
                tmp.append(r.val)
                if r.left:
                    queue.append(r.left)
                if r.right:
                    queue.append(r.right)
            # 将临时list加入最终返回结果中
            res.append(tmp)
        return res


class Solution31:
    """
    从前序与中序遍历序列构造二叉树
    https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。
    """

    @staticmethod
    def build_tree(pre_order: List[int], in_order: List[int]) -> TreeNode:
        def helper(in_left, in_right):
            # 如果这里没有节点构造二叉树了，就结束
            if in_left > in_right:
                return None

            # 选择 post_idx 位置的元素作为当前子树根节点
            val = pre_order.pop(0)
            root = TreeNode(val)

            # 根据 root 所在位置分成左右两棵子树
            index = idx_map[val]

            # 构造左子树
            root.left = helper(in_left, index - 1)

            # 构造右子树
            root.right = helper(index + 1, in_right)

            return root

        # 建立（元素，下标）键值对的哈希表
        idx_map = {val: idx for idx, val in enumerate(in_order)}
        return helper(0, len(in_order) - 1)

class Solution32:
    """
    将有序数组转换为二叉搜索树
    https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
    给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
    高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
    """

    @staticmethod
    def sorted_array_to_bst(nums: List[int]) -> TreeNode:
        def helper(left, right):
            if left > right:
                return None

            # 总是选择中间位置左边的数字作为根节点
            mid = (left + right) // 2
            # mid = (left + right + 1) // 2 也可以总是选择中间位置右边的数字作为根节点
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root

        return helper(0, len(nums) - 1)

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution33:
    """
    有序链表转换二叉搜索树
    https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/
    给定一个单链表的头节点  head ，其中的元素 按升序排序 ，将其转换为高度平衡的二叉搜索树。
    本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差不超过 1。
    """

    @staticmethod
    def sorted_list_to_bst(head: Optional[ListNode]) -> Optional[TreeNode]:
        def helper(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root

        nums = []
        while head:
            nums.append(head.val)
            head = head.next
        return helper(0, len(nums) - 1)

    @staticmethod
    def sorted_list_to_bst_v2(head: ListNode) -> TreeNode:
        def get_median(left: ListNode, right: ListNode) -> ListNode:
            fast = slow = left
            while fast != right and fast.next != right:
                fast = fast.next.next
                slow = slow.next
            return slow

        def build_tree(left: ListNode, right: ListNode) -> TreeNode:
            if left == right:
                return None
            mid = get_median(left, right)
            root = TreeNode(mid.val)
            root.left = build_tree(left, mid)
            root.right = build_tree(mid.next, right)
            return root

        return build_tree(head, None)


class Solution34:
    """
    平衡二叉树
    https://leetcode-cn.com/problems/balanced-binary-tree/
    给定一个二叉树，判断它是否是高度平衡的二叉树。
    一棵高度平衡二叉树定义为：一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
    """

    def is_balanced(self, root: TreeNode) -> bool:
        def height(root: TreeNode) -> int:
            if not root:
                return 0
            return max(height(root.left), height(root.right)) + 1

        if not root:
            return True
        return abs(height(root.left) - height(root.right)) <= 1 and self.is_balanced(root.left) and self.is_balanced(root.right)

    @staticmethod
    def is_balanced_v2(root: TreeNode) -> bool:
        def height(root: TreeNode) -> int:
            if not root:
                return 0
            left_height = height(root.left)
            right_height = height(root.right)
            if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
                return -1
            else:
                return max(left_height, right_height) + 1

        return height(root) >= 0

class Solution35:
    """
    二叉树展开为链表
    https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/
    给你二叉树的根结点 root ，请你将它展开为一个单链表：
    展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
    展开后的单链表应该与二叉树 先序遍历 顺序相同。
    """

    @staticmethod
    def flatten(root: TreeNode) -> None:
        pre_order_list = list()
        def pre_order_traversal(root: TreeNode):
            if root:
                pre_order_list.append(root)
                pre_order_traversal(root.left)
                pre_order_traversal(root.right)

        pre_order_traversal(root)
        size = len(pre_order_list)
        for i in range(1, size):
            prev, curr = pre_order_list[i - 1], pre_order_list[i]
            prev.left = None
            prev.right = curr


class Solution36:
    """
    不同的子序列
    https://leetcode-cn.com/problems/distinct-subsequences/
    给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
    字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
    题目数据保证答案符合 32 位带符号整数范围。
    输入：s = "rabbbit", t = "rabbit"
    输出：3
    解释：
    如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
    rabbbit
    rabbbit
    rabbbit
    """

    @staticmethod
    def num_distinct(s: str, t: str) -> int:
        @functools.lru_cache  # Python functools 记忆化模块，等价于哈希表记录的功能
        def dfs(i, j):
            if j == len(t):  # t匹配完了
                return 1

            if i == len(s):  # s匹配完了依然没找到答案
                return 0

            cnt = 0
            # 跳过s[i]，并进行下一步搜索：s[i+1]与t[j]匹配

            cnt += dfs(i + 1, j)

            # 选择s[i]，并进行下一步搜索
            if s[i] == t[j]:  # 能选择s[i]的前提条件为：s[i] == t[j]
                cnt += dfs(i + 1, j + 1)
            return cnt

        return dfs(0, 0)  # 初始从s[0]和t[0]开始搜索

    @staticmethod
    def num_distinct_v2(s: str, t: str) -> int:
        m, n = len(s), len(t)
        if m < n:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维数组 dp[i][j] 表示在 s[i:] 的子序列中 t[j:] 出现的个数。s[i:] 表示 s 从下标 i 到末尾的子字符串，t[j:] 表示 t 从下标 j 到末尾的子字符串

        for i in range(m + 1):  # 当 j=n 时, t[j:] 为空字符串，由于空字符串是任何字符串的子序列，因此对任意 0≤ i ≤m，有 dp[i][n]=1
            dp[i][n] = 1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:  # 当 s[i]=t[j] 时，如果 s[i] 和 t[j] 匹配，则考虑 t[j+1:] 作为 s[i+1:] 的子序列，子序列数为 dp[i+1][j+1]，
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j]  # 如果 s[i] 不和 t[j] 匹配，则考虑 t[j:] 作为 s[i+1:] 的子序列，子序列数为 dp[i+1][j]。
                else:
                    dp[i][j] = dp[i + 1][j]  # 当 s[i]!=t[j] 时，s[i] 不能和 t[j] 匹配，因此只考虑 t[j:] 作为 s[i+1:] 的子序列，子序列数为 dp[i+1][j]。

        return dp[0][0]


class Solution37:
    """
    三角形最小路径和
    https://leetcode-cn.com/problems/triangle/
    给定一个三角形 triangle ，找出自顶向下的最小路径和。
    每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
    也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
    """

    @staticmethod
    def minimum_total(triangle: List[List[int]]) -> int:
        n = len(triangle)
        f = [[0] * n for _ in range(n)]
        f[0][0] = triangle[0][0]

        for i in range(1, n):
            f[i][0] = f[i - 1][0] + triangle[i][0]  # 每一行的第一列只能由上一行的第一列转换而来

            for j in range(1, i):
                f[i][j] = min(f[i - 1][j - 1], f[i - 1][j]) + triangle[i][j]  # 从上一行 j-1 列和 j 列中取值小的来转换

            f[i][i] = f[i - 1][i - 1] + triangle[i][i]  # # 每一行的最后列只能由上一行的最后一列转换而来

        return min(f[n - 1])


class Solution38:
    """
    https://leetcode.cn/problems/next-greater-node-in-linked-list/
    链表中的下一个更大节点
    给定一个长度为 n 的链表 head
    对于列表中的每个节点，查找下一个 更大节点 的值。也就是说，对于每个节点，找到它旁边的第一个节点的值，这个节点的值 严格大于 它的值。
    """

    @staticmethod
    def next_larger_nodes(head: Optional[ListNode]) -> List[int]:
        temp = []
        while head:
            temp.append(head.val)
            head = head.next
        st = []  # 单调栈，基于当前链表建一个数组，设一个单调栈存取数组下标，栈底元素最大，每次遍历都将小于当前元素的栈顶元素弹出并且更新答案数组
        n = len(temp)
        ans = [0] * n
        for i in range(n):
            while len(st) != 0 and temp[st[-1]] < temp[i]:
                ans[st[- 1]] = temp[i]
                st.pop()
            st.append(i)
        return ans


class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.add_to_head(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.remove_tail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.move_to_head(node)

    def add_to_head(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    @staticmethod
    def remove_node(node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def move_to_head(self, node):
        self.remove_node(node)
        self.add_to_head(node)

    def remove_tail(self):
        node = self.tail.prev
        self.move_to_head(node)
        return node


class Solution39:
    """
    买卖股票的最佳时机 II
    https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/
    给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
    在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。
    返回 你能获得的 最大 利润 。


    输入：prices = [7,1,5,3,6,4]
    输出：7
    解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
    随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
    总利润为 4 + 3 = 7 。
    """

    @staticmethod
    def max_profit(prices: List[int]) -> int:
        dp = [0] * len(prices)
        for i in range(1, len(prices)):
            new_price = dp[i - 1] + (prices[i] - prices[i - 1])
            dp[i] = max(dp[i - 1], new_price)
        return dp[len(prices) - 1]

    @staticmethod
    def max_profit_v2(prices: List[int]) -> int:
        n = len(prices)
        # dp[i][0] : 第i天结束时，手上没有股票的最大利润
        # dp[i][1] : 第i天结束时，手上持有股票的最大利润
        dp = [[0] * 2 for _ in range(n)]

        # 初始化
        dp[0][0] = 0
        dp[0][1] = -prices[0]

        for i in range(1, n):
            # dp[i][0]：（前一天没有股票，前一天持有股票+当天卖出）的最大值
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            # dp[i][1]：（前一天持有股票，前一天没有股票+当天买入）的最大值
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])

        return dp[n - 1][0]


class Solution40:
    """
    买卖股票的最佳时机 III
    https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/
    给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

    设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

    输入：prices = [3,3,5,0,0,3,1,4]
    输出：6
    解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
         随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
    """

    @staticmethod
    def max_profit(prices: List[int]) -> int:
        # buy1 只进行过一次买操作；
        # sell1 进行了一次买操作和一次卖操作，即完成了一笔交易；
        # buy2 在完成了一笔交易的前提下，进行了第二次买操作；
        # sell2 完成了全部两笔交易。
        n = len(prices)
        buy1 = buy2 = -prices[0]
        sell1 = sell2 = 0
        for i in range(1, n):
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        return sell2


class Solution41:
    """
     二叉树中的最大路径和
    https://leetcode.cn/problems/binary-tree-maximum-path-sum/
    路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
    路径和 是路径中各节点值的总和。
    给你一个二叉树的根节点 root ，返回其 最大路径和 。
    """

    def __init__(self):
        self.max_sum = float("-inf")

    def max_path_sum(self, root: TreeNode):
        def max_gain(node):
            if not node:
                return 0

            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            price_new_path = node.val + left_gain + right_gain

            # 更新答案
            self.max_sum = max(self.max_sum, price_new_path)

            # 返回节点的最大贡献值
            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return self.max_sum


class Solution42:
    """
    最长连续序列
    https://leetcode.cn/problems/longest-consecutive-sequence/
    给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
    """

    @staticmethod
    def longest_consecutive(nums: List[int]) -> int:
        longest_streak = 0
        num_set = set(nums)
        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                longest_streak = max(longest_streak, current_streak)
        return longest_streak


class Solution43(object):
    """
    分割回文串
    https://leetcode.cn/problems/palindrome-partitioning/
    给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
    """

    def partition(self, s):
        res = []
        self.backtrack(s, res, [])
        return res

    def backtrack(self, s, res, path):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s) + 1):  # 注意起始和结束位置
            if s[:i] == s[:i][::-1]:
                self.backtrack(s[i:], res, path + [s[:i]])

    @staticmethod
    def partition_v2(s: str) -> List[List[str]]:
        result = []
        path = []

        # 判断是否是回文串
        def pending_s(s):
            l, r = 0, len(s) - 1
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        # 回溯函数，这里的index作为遍历到的索引位置，也作为终止判断的条件
        def back_track(s, index):
            # 如果对整个字符串遍历完成，并且走到了这一步，则直接加入result
            if index == len(s):
                result.append(path[:])
                return

            # 遍历每个子串
            for i in range(index, len(s)):
                # 剪枝，因为要求每个元素都是回文串，那么我们只对回文串进行递归，不是回文串的部分直接不care它
                # 当前子串是回文串
                if pending_s(s[index: i + 1]):
                    # 加入当前子串到path
                    path.append(s[index: i + 1])
                    # 从当前i+1处重复递归
                    back_track(s, i + 1)
                    # 回溯
                    path.pop()

        back_track(s, 0)
        return result


class Solution44:
    """
    分割回文串 II
    https://leetcode.cn/problems/palindrome-partitioning-ii/
    给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。
    返回符合要求的 最少分割次数 。
    """

    @staticmethod
    def min_cut(s: str):
        l = len(s)
        f = [i for i in range(l)]  # 前i个元素最少的分割次数
        for i in range(l):
            for j in range(i + 1):  # 对于第i个元素，遍历之前的所有组合进行判断
                tmp = s[j:i + 1]
                if tmp == tmp[::-1]:  # s[j:i+1]是回文
                    f[i] = min(f[i], f[j - 1] + 1) if j > 0 else 0
        return f[l - 1]


class Solution45:
    """
    分发糖果
    https://leetcode.cn/problems/candy/
    n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
    你需要按照以下要求，给这些孩子分发糖果：

    每个孩子至少分配到 1 个糖果。
    相邻两个孩子评分更高的孩子会获得更多的糖果。
    请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。

    输入：ratings = [1,0,2]
    输出：5
    解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
    """

    @staticmethod
    def candy(ratings: List[int]) -> int:
        n = len(ratings)
        left = [0] * n
        for i in range(n):
            if i > 0 and ratings[i] > ratings[i - 1]:
                left[i] = left[i - 1] + 1
            else:
                left[i] = 1

        right = ret = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and ratings[i] > ratings[i + 1]:
                right += 1
            else:
                right = 1
            ret += max(left[i], right)

        return ret


class Solution46:
    """
    单词拆分
    给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
    不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
    https://leetcode.cn/problems/word-break/
    """

    @staticmethod
    def word_break(s: str, word_dict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n + 1)  # dp[i] 表示 s 的前 i 位是否可以用 wordDict 中的单词表示。
        dp[0] = True  # 初始化 dp[0]=True，空字符可以被表示。
        for i in range(n):  # 遍历字符串的所有子串
            for j in range(i + 1, n + 1):
                if dp[i] and (s[i:j] in word_dict):  # dp[i]=True 说明 s 的前 i 位可以用 wordDict 表示, s[i:j]出现在 wordDict 中，说明 s 的前 j 位可以表示。
                    dp[j] = True
        return dp[-1]


class Solution47:
    """
    单词拆分 II
    https://leetcode.cn/problems/word-break-ii/
    给定一个字符串 s 和一个字符串字典 wordDict ，在字符串 s 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。以任意顺序 返回所有这些可能的句子。
    词典中的同一个单词可能在分段中被重复使用多次。
    """

    def word_break(self, s: str, word_dict: List[str]) -> List[str]:
        res = []
        self.backtrack(s, res, [], word_dict)
        return res

    def backtrack(self, s, res, path, word_dict):
        if not s:
            res.append(" ".join(path))
            return res
        for i in range(1, len(s) + 1):  # 注意起始和结束位置
            if s[:i] in word_dict:
                self.backtrack(s[i:], res, path + [s[:i]], word_dict)



class Solution48:
    """
    重排链表
    https://leetcode.cn/problems/reorder-list/
    给定一个单链表 L 的头节点 head ，单链表 L 表示为：
    输入：head = [1,2,3,4,5]
    输出：[1,5,2,4,3]
    """

    @staticmethod
    def reorder_list(head: ListNode) -> None:
        if not head:
            return

        vec = list()
        node = head
        while node:
            vec.append(node)
            node = node.next

        i, j = 0, len(vec) - 1
        while i < j:
            vec[i].next = vec[j]
            i += 1
            if i == j:
                break
            vec[j].next = vec[i]
            j -= 1

        vec[i].next = None

class Solution49:
    """
    归并排序链表
    https://leetcode.cn/problems/sort-list/
    给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

    """

    @staticmethod
    def sort_list(head: ListNode) -> ListNode:
        def sort_func(head: ListNode, tail: ListNode) -> ListNode:
            if not head:  # 找到链表的中点，以中点为分界，将链表拆分成两个子链表。寻找链表的中点可以使用快慢指针的做法，
                return head # 快指针每次移动 2 步，慢指针每次移动 1 步，当快指针到达链表末尾时，慢指针指向的链表节点即为链表的中点。
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            left = sort_func(head, mid)  # 对两个子链表分别排序。
            right = sort_func(mid, tail)
            return merge(left, right)

        def merge(head1: ListNode, head2: ListNode) -> ListNode: # 将两个排序后的子链表合并，得到完整的排序后的链表。
            dummy_head = ListNode(0)
            temp, temp1, temp2 = dummy_head, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummy_head.next

        return sort_func(head, None)
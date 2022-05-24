# -*- coding: utf-8 -*-
import sys
import heapq
import itertools
import functools
import collections
from collections import Counter
from typing import List, Optional


class Solution1:
    """
    51. N 皇后
    https://leetcode-cn.com/problems/n-queens/
    n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
    给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
    每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
    """

    def __init__(self):
        self.count = 0
        self.out_put = []

    def get_out_put(self):
        column_list = []
        for row in range(0, self.n):
            column_str = ""
            for column in range(0, self.n):
                if self.result[row] == column:
                    column_str += "Q"
                else:
                    column_str += "."
            column_list.append("%s" % column_str)
        return column_list

    def is_column_ok(self, row, column):
        left_column = column - 1
        right_column = column + 1
        for i in range(row - 1, -1, -1):  # 逐行往上考察
            if self.result[i] in [column, left_column, right_column]:  # 检查上一行,对角线上, 右上对角线上是否有皇后
                return False
            left_column -= 1
            right_column += 1
        return True

    def cal_8_queens(self, row):
        if row == self.n:  # n个棋子都放置好了，打印结果
            self.count += 1
            ot = self.get_out_put()
            self.out_put.append(ot)
            return

        for column in range(0, self.n):  # 每一行有 n个放法
            if self.is_column_ok(row, column):
                self.result[row] = column  # 第 row 行的棋子放在 column 列
                self.cal_8_queens(row + 1)  # 计算下一个

    def solve_n_queens(self, n: int) -> List[List[str]]:
        self.n = n
        self.result = [None] * n
        self.cal_8_queens(0)
        return self.out_put


class Solution2:
    """
    最大子数组和
    https://leetcode-cn.com/problems/maximum-subarray/
    给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    子数组 是数组中的一个连续部分。

    定义状态（定义子问题） dp[i]：表示以 nums[i] 结尾 的 连续 子数组的最大和
    根据状态的定义，由于 nums[i] 一定会被选取，并且以 nums[i] 结尾的连续子数组与以 nums[i - 1] 结尾的连续子数组只相差一个元素 nums[i] 。
    假设数组 nums 的值全都严格大于 0，那么一定有 dp[i] = dp[i - 1] + nums[i]。
    可是 dp[i - 1] 有可能是负数，于是分类讨论：
    如果 dp[i - 1] > 0，那么可以把 nums[i] 直接接在 dp[i - 1] 表示的那个数组的后面，得到和更大的连续子数组；
    如果 dp[i - 1] <= 0，那么 nums[i] 加上前面的数 dp[i - 1] 以后值不会变大。于是 dp[i] 「另起炉灶」，此时单独的一个 nums[i] 的值，就是 dp[i]。
    """

    @staticmethod
    def max_sub_array(nums: List[int]) -> int:
        size = len(nums)
        if size == 0:
            return 0
        dp = [0 for _ in range(size)]

        dp[0] = nums[0]
        for i in range(1, size):
            if dp[i - 1] >= 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)

    @staticmethod
    def max_sub_array_v2(nums: List[int]) -> int:
        """nums = [-2,1,-3,4,-1,2,1,-5,4]"""
        size = len(nums)
        pre = 0
        res = nums[0]
        for i in range(size):
            pre = max(nums[i], pre + nums[i])
            res = max(res, pre)
        return res


class Solution3:
    """
    螺旋矩阵
    https://leetcode-cn.com/problems/spiral-matrix/
    给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
    """

    @staticmethod
    def spiral_order(matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # 初始化
        x, y = 0, 0
        dx, dy = dirs.pop(0)
        dirs.append((dx, dy))
        res = []

        # 逆时针循环
        while len(res) < m * n:
            res.append(matrix[x][y])
            matrix[x][y] = -101
            if not (0 <= x + dx <= m - 1 and 0 <= y + dy <= n - 1 and matrix[x + dx][y + dy] != -101):
                dx, dy = dirs.pop(0)
                dirs.append((dx, dy))
            x, y = x + dx, y + dy
        return res


class Solution4:
    """
    合并区间
    https://leetcode-cn.com/problems/merge-intervals/
    以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
    输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
    输出：[[1,6],[8,10],[15,18]]
    解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

    如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。
    """

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged


class Solution5:
    """
    https://leetcode-cn.com/problems/spiral-matrix-ii/
    螺旋矩阵 II
    给你一个正整数 n ，生成一个包含 1 到 n的平方的所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
    """
    @staticmethod
    def generate_matrix(n: int) -> List[List[int]]:
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = 0, 0
        dx, dy = dirs.pop(0)
        dirs.append((dx, dy))
        matrix = [[None] * n for _ in range(n)]
        i = 1
        while i < n * n + 1:
            matrix[x][y] = i
            if not (0 <= x + dx <= n - 1 and 0 <= y + dy <= n - 1 and not matrix[x + dx][y + dy]):
                dx, dy = dirs.pop(0)
                dirs.append((dx, dy))
            x, y = x + dx, y + dy
            i += 1
        return matrix


class Solution6:
    """
    https://leetcode-cn.com/problems/permutation-sequence/
    排列序列
    给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。
    """

    @staticmethod
    def get_permutation(n: int, k: int) -> str:
        aa = "".join([str(x) for x in range(1, n + 1)])
        resp = []
        bb = itertools.permutations(aa, n)
        for x in bb:
            resp.append("".join(list(x)))
        return resp[k - 1]


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution7:
    """
    61. 旋转链表
    https://leetcode-cn.com/problems/rotate-list/
    给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
    """

    @staticmethod
    def rotate_right(head: Optional[ListNode], k: int) -> Optional[ListNode]:
        i = 0
        count = 1
        if not head:
            return head

        temp = head
        while temp and temp.next:
            count += 1
            temp = temp.next

        k = k % count

        while i < k:
            i += 1
            cur = head
            pre = None
            while cur and cur.next:
                pre = cur
                cur = cur.next
            if pre:
                pre.next = None
                cur.next = head
                head = cur
        return head


class Solution8:
    """
    不同路径
    https://leetcode-cn.com/problems/unique-paths/
    一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
    机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
    问总共有多少条不同的路径？

    我们用 f(i,j) 表示从左上角走到 (i,j) 的路径数量，其中 i 和 j 的范围分别是 [0,m) 和 [0,n)。
    由于我们每一步只能从向下或者向右移动一步，因此要想走到 (i,j)，如果向下走一步，那么会从 (i−1,j) 走过来；
    如果向右走一步，那么会从 (i,j−1) 走过来。
    """

    @staticmethod
    def unique_paths(m: int, n: int) -> int:
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]  # 第一行和第一例默认为1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]


class Solution9:
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

    @staticmethod
    def unique_paths_with_obstacles_v2(obstacle_grid: List[List[int]]) -> int:
        m, n = len(obstacle_grid), len(obstacle_grid[0])
        mem = [1] + [0] * (n - 1)
        for i in range(m):
            for j in range(n):
                if obstacle_grid[i][j] == 1:
                    mem[j] = 0
                elif j - 1 >= 0:
                    mem[j] += mem[j - 1]
        return mem[-1]


class Solution10:
    """
    最小路径和
    https://leetcode-cn.com/problems/minimum-path-sum/
    给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
    由于路径的方向只能是向下或向右，因此网格的第一行的每个元素只能从左上角元素开始向右移动到达，网格的第一列的每个元素只能从左上角元素开始向下移动到达，此时的路径是唯一的，因此每个元素对应的最小路径和即为对应的路径上的数字总和。
    对于不在第一行和第一列的元素，可以从其上方相邻元素向下移动一步到达，或者从其左方相邻元素向右移动一步到达，
    元素对应的最小路径和等于其上方相邻元素与其左方相邻元素两者对应的最小路径和中的最小值加上当前元素的值。
    由于每个元素对应的最小路径和与其相邻元素对应的最小路径和有关，因此可以使用动态规划求解。
    """

    @staticmethod
    def min_path_sum(grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, columns = len(grid), len(grid[0])
        dp = [[0] * columns for _ in range(rows)]
        dp[0][0] = grid[0][0]

        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] + grid[i][0]  # 第一列的每个元素只能从左上角元素开始向下移动到达

        for j in range(1, columns):
            dp[0][j] = dp[0][j - 1] + grid[0][j]  # 第一行的每个元素只能从左上角元素开始向右移动到达

        for i in range(1, rows):  # 不在第一行和第一列的元素，最小路径和等于其上方相邻元素与其左方相邻元素两者对应的最小路径和中的最小值加上当前元素的值。
            for j in range(1, columns):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[rows - 1][columns - 1]


class Solution11:
    """
    有效数字
    https://leetcode-cn.com/problems/valid-number/

    有效数字（按顺序）可以分成以下几个部分：

    一个 小数 或者 整数
    （可选）一个 'e' 或 'E' ，后面跟着一个 整数
    小数（按顺序）可以分成以下几个部分：

    （可选）一个符号字符（'+' 或 '-'）
    下述格式之一：
    至少一位数字，后面跟着一个点 '.'
    至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
    一个点 '.' ，后面跟着至少一位数字
    整数（按顺序）可以分成以下几个部分：

    （可选）一个符号字符（'+' 或 '-'）
    至少一位数字

    部分有效数字列举如下：["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]

    部分无效数字列举如下：["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]

    1:初始状态 (空字符串或者纯空格)
    2:符号位
    3:数字位 (形如-164,可以作为结束)
    4:小数点
    5:小数点后的数字(形如.721或者-123.6,可以作为结束)
    6:指数e
    7:指数后面的符号位
    8:指数后面的数字(形如+1e-6,可以作为结束)
    9:状态3,5,8后面多了空格(主要为了判断"1 1"是不合理的)

    """

    def isNumber(self, s: str) -> bool:
        # DFA transitions: dict[action] -> successor
        states = [{},
                  # state 1
                  {"blank": 1, "sign": 2, "digit": 3, "dot": 4},
                  # state 2
                  {"digit": 3, "dot": 4},
                  # state 3
                  {"digit": 3, "dot": 5, "e|E": 6, "blank": 9},
                  # state 4
                  {"digit": 5},
                  # state 5
                  {"digit": 5, "e|E": 6, "blank": 9},
                  # state 6
                  {"sign": 7, "digit": 8},
                  # state 7
                  {"digit": 8},
                  # state 8
                  {"digit": 8, "blank": 9},
                  # state 9
                  {"blank": 9}]

        def strToAction(st):
            if '0' <= st <= '9':
                return "digit"
            if st in "+-":
                return "sign"
            if st in "eE":
                return "e|E"
            if st == '.':
                return "dot"
            if st == ' ':
                return "blank"
            return None

        currState = 1
        for c in s:
            action = strToAction(c)
            if action not in states[currState]:
                return False
            currState = states[currState][action]

        # ending states: 3,5,8,9
        return currState in {3, 5, 8, 9}


class Solution12:
    """
    加一
    https://leetcode-cn.com/problems/plus-one/

    给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
    最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
    你可以假设除了整数 0 之外，这个整数不会以零开头。
    """

    @staticmethod
    def plus_one(digits: List[int]) -> List[int]:
        n = len(digits)
        for i in range(n - 1, -1, -1):  # 对数组进行一次逆序遍历，找出第一个不为 9 的元素，将其加一并将后续所有元素置零即可
            if digits[i] != 9:
                digits[i] += 1
                for j in range(i + 1, n):
                    digits[j] = 0
                return digits

        # digits 中所有的元素均为 9
        return [1] + [0] * n


# blank 返回长度为 n 的由空格组成的字符串
def blank(n: int) -> str:
    return ' ' * n


class Solution13:
    """
    文本左右对齐
    https://leetcode-cn.com/problems/text-justification/
    给定一个单词数组 words 和一个长度 maxWidth ，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。

    你应该使用 “贪心算法” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。

    要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

    文本的最后一行应为左对齐，且单词之间不插入额外的空格。
    单词是指由非空格字符组成的字符序列。
    每个单词的长度大于 0，小于等于 maxWidth。
    输入单词数组 words 至少包含一个单词。
    """

    @staticmethod
    def fullJustify(words: List[str], maxWidth: int) -> List[str]:
        ans = []
        right, n = 0, len(words)
        while True:
            left = right  # 当前行的第一个单词在 words 的位置
            sumLen = 0  # 统计这一行单词长度之和
            # 循环确定当前行可以放多少单词，注意单词之间应至少有一个空格
            while right < n and sumLen + len(words[right]) + right - left <= maxWidth:
                sumLen += len(words[right])
                right += 1

            # 当前行是最后一行：单词左对齐，且单词之间应只有一个空格，在行末填充剩余空格
            if right == n:
                s = " ".join(words[left:])
                ans.append(s + blank(maxWidth - len(s)))
                break

            numWords = right - left
            numSpaces = maxWidth - sumLen

            # 当前行只有一个单词：该单词左对齐，在行末填充空格
            if numWords == 1:
                ans.append(words[left] + blank(numSpaces))
                continue

            # 当前行不只一个单词
            avgSpaces = numSpaces // (numWords - 1)
            extraSpaces = numSpaces % (numWords - 1)
            s1 = blank(avgSpaces + 1).join(words[left:left + extraSpaces + 1])  # 拼接额外加一个空格的单词
            s2 = blank(avgSpaces).join(words[left + extraSpaces + 1:right])  # 拼接其余单词
            ans.append(s1 + blank(avgSpaces) + s2)

        return ans


class Solution14:
    """
    爬楼梯
    https://leetcode-cn.com/problems/climbing-stairs/
    假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    """

    @staticmethod
    def climb_stairs(n: int) -> int:
        # 找到规律db[i] = db[i-1] + db[i-2]
        # db[0] = 1 , db[1] = 1, 零阶和一阶都是一次
        # db[]的长度为n+1，因为列表头多了一个零阶
        db = [0 for _ in range(n + 1)]
        db[0] = db[1] = 1
        for i in range(2, n + 1):
            db[i] = db[i - 1] + db[i - 2]
        return db[-1]


class Solution15:
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


class Solution16:
    """
    颜色分类
    https://leetcode-cn.com/problems/sort-colors/
    给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

    我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
    必须在不使用库的sort函数的情况下解决这个问题。

    """

    @staticmethod
    def sort_colors(nums: List[int]) -> None:
        n = len(nums)
        p0, p2 = 0, n - 1
        i = 0
        while i <= p2:
            while i <= p2 and nums[i] == 2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
            i += 1


class Solution17:
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




class Solution18:
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


class Solution19:
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

    @staticmethod
    def subsets_v3(nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)

        def helper(i, tmp):
            res.append(tmp)
            for j in range(i, n):
                helper(j + 1, tmp + [nums[j]])

        helper(0, [])
        return res


class Solution20:
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


class Solution21:
    """
    删除有序数组中的重复项 II
    https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/
    给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。
    不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
    nums = [1,1,1,2,2,3]
    """

    @staticmethod
    def remove_duplicates(nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n

        slow, fast = 2, 2  # 数组的前两个数必然可以被保留
        while fast < n:
            # 检查上上个应该被保留的元素nums[slow−2]是否和当前待检查元素nums[fast]相同
            if nums[slow - 2] != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        return slow  # 从nums[0]到nums[slow−1]的每个元素都不相同

    @staticmethod
    def remove_duplicates_v2(nums: List[int]) -> int:

        def solve(k):  # 最多保留k位相同数字
            slow = 0  # 慢指针从0开始
            for fast in nums:  # 快指针遍历整个数组
                # 检查被保留的元素nums[slow−k]是否和当前待检查元素fast相同
                if slow < k or nums[slow - k] != fast:
                    nums[slow] = fast
                    slow += 1
            return slow  # 从nums[0]到nums[slow−1]的每个元素都不相同

        return solve(2)


class Solution22:
    """
    搜索旋转排序数组 II
    https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/
    已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
    在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
    给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
    你必须尽可能减少整个操作步骤。
    """

    @staticmethod
    def search(nums: List[int], target: int) -> bool:
        if not nums:
            return False

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = low + (high - low) // 2
            # 如果找到，返回结果
            if nums[mid] == target:
                return True
            # 相当于去重
            if nums[low] == nums[mid] == nums[high]:
                low += 1
                high -= 1
            # 如果左区间有序
            elif nums[low] <= nums[mid]:
                # target 在左区间
                if nums[low] <= target < nums[mid]:
                    high = mid - 1
                # target 在右区间
                else:
                    low = mid + 1
            # 如果右区间有序
            else:
                # target 在右区间
                if nums[mid] < target <= nums[high]:
                    low = mid + 1
                # target 在左区间
                else:
                    high = mid - 1

        return False


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution23:
    """
    82. 删除排序链表中的重复元素 II
    https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
    给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
    """

    @staticmethod
    def delete_duplicates(head: ListNode) -> ListNode:
        if not head:
            return head

        dummy = ListNode(0, head)

        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                x = cur.next.val
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next

        return dummy.next


class Solution24:
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

    @staticmethod
    def largest_rectangle_area_v2(heights: List[int]) -> int:
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


class Solution25:
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


class Solution26:
    """
    分隔链表
    https://leetcode-cn.com/problems/partition-list/
    给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。
    你应当 保留 两个分区中每个节点的初始相对位置。
    """

    @staticmethod
    def partition(head: ListNode, x: int) -> ListNode:
        small = ListNode()
        small_cur = small
        big = ListNode()
        big_cur = big
        while head:
            node = ListNode(head.val)
            if head.val < x:
                small_cur.next = node
                small_cur = small_cur.next
            else:
                big_cur.next = node
                big_cur = big_cur.next
            head = head.next
        small_cur.next = big.next
        return small.next


class Solution27:
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


class Solution28:
    """
    https://leetcode-cn.com/problems/merge-sorted-array/
    给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
    请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
    最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
    """

    @staticmethod
    def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1[m:] = nums2
        nums1.sort()


class Solution29:
    """
    格雷编码
    https://leetcode-cn.com/problems/gray-code/

    n 位格雷码序列 是一个由 2n 个整数组成的序列，其中：
    每个整数都在范围 [0, 2n - 1] 内（含 0 和 2n - 1）
    第一个整数是 0
    一个整数在序列中出现 不超过一次
    每对 相邻 整数的二进制表示 恰好一位不同 ，且
    第一个 和 最后一个 整数的二进制表示 恰好一位不同
    给你一个整数 n ，返回任一有效的 n 位格雷码序列 。
    """

    @staticmethod
    def gray_code(n: int) -> List[int]:
        res, head = [0], 1
        for i in range(n):
            for j in res[::-1]:  # 对上一轮结果逆序遍历，对每个数据的二进制最前面补 1
                res.append(head + j)
            head <<= 1
        return res


class Solution30(object):
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


class Solution31:
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


class Solution32:
    """
    反转链表 II
    https://leetcode-cn.com/problems/reverse-linked-list-ii/
    给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表
    输入：head = [1,2,3,4,5], left = 2, right = 4
    输出：[1,4,3,2,5]
    """

    @staticmethod
    def reverse_between(head: ListNode, left: int, right: int) -> ListNode:

        stack = []
        index = 0
        left_node = None
        right_node = None
        dump_head = ListNode(val=-1, next=head)
        cur = dump_head
        while cur and index <= right:
            if index + 1 == left:
                left_node = cur
            if index >= left:
                stack.append(cur.val)
            cur = cur.next
            right_node = cur
            index += 1

        while stack:
            node = ListNode(val=stack.pop())
            left_node.next = node
            left_node = node

        left_node.next = right_node
        return dump_head.next


class Solution33:
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

        def get_ip(ip: list, idx):
            # 4. 长度大于 4 剪枝
            if len(ip) == 4:
                # 5. 正好取完，才对
                if idx == len(s):
                    rst.append('.'.join(ip))
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
                ip.append(sub)
                get_ip(ip, idx + i)
                ip.pop()

        get_ip([], 0)
        return rst


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution34:
    """
    二叉树的中序遍历
    https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
    给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
    """

    @staticmethod
    def in_order_traversal(root: Optional[TreeNode]) -> List[int]:
        def traversal(sub_root, order):
            if not sub_root:
                return
            traversal(sub_root.left, order)
            order.append(sub_root.val)
            traversal(sub_root.right, order)

        _order = []
        traversal(root, _order)
        return _order

    @staticmethod
    def in_order_traversal_v2(root: Optional[TreeNode]) -> List[int]:
        stack = []
        ret = []
        stack.append((0, root))

        while len(stack) != 0:
            op, node = stack.pop()
            if node is None:
                continue
            if op == 1:
                ret.append(node.val)
            else:
                stack.append((0, node.right))
                stack.append((1, node))
                stack.append((0, node.left))
        return ret


import functools


class Solution35:
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


class Solution36:
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

    @functools.lru_cache(None)
    def num_trees_v2(self, n: int) -> int:
        if n in [0, 1]:
            return 1
        res = 0
        for i in range(1, n + 1):
            left_tree_num = self.num_trees_v2(i - 1)
            right_tree_num = self.num_trees_v2(n - i)
            res += left_tree_num * right_tree_num
        return res


class Solution37:
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


class Solution38:
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


class Solution39(object):
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


class Solution40:
    """
    相同的树
    https://leetcode-cn.com/problems/same-tree/
    给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。
    如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
    """

    def is_same_tree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        elif not p or not q:
            return False
        elif p.val != q.val:
            return False
        return self.is_same_tree(p.left, q.left) and self.is_same_tree(p.right, q.right)


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


class Solution41:
    """
    对称二叉树
    https://leetcode-cn.com/problems/symmetric-tree/
    给你一个二叉树的根节点 root ， 检查它是否轴对称。
    """

    def is_symmetric(self, root: TreeNode) -> bool:
        return self.check(root, root)

    def check(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.check(p.left, q.right) and self.check(p.right, q.left)


class Solution42(object):
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


class Solution43:
    """
    二叉树的锯齿形层序遍历
    https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/
    给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
    """

    @staticmethod
    def zig_level_order(root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        queue = [root]
        flag = 1
        while queue:
            q_size = len(queue)
            temp = []
            for _ in range(q_size):
                r = queue.pop(0)
                temp.append(r.val)
                if r.left:
                    queue.append(r.left)
                if r.right:
                    queue.append(r.right)
            if flag == 1:
                res.append(temp)
                flag = -1
            else:
                res.append(temp[::-1])
                flag = 1
        return res


class Solution44:
    """
    二叉树的最大深度
    https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/
    给定一个二叉树，找出其最大深度。
    二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
    """

    def max_depth(self, root):  # 深度优先
        if root is None:
            return 0
        else:
            left_height = self.max_depth(root.left)
            right_height = self.max_depth(root.right)
            return max(left_height, right_height) + 1

    @staticmethod
    def max_depth(root: Optional[TreeNode]) -> int:  # 广度优先
        if not root:
            return 0
        res = 0
        queue = [root]
        while queue:
            q_size = len(queue)
            for _ in range(q_size):
                x = queue.pop(0)
                if x.left:
                    queue.append(x.left)
                if x.right:
                    queue.append(x.right)
            res += 1
        return res


class Solution45:
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


class Solution46:
    """
    从中序与后序遍历序列构造二叉树
    https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
    给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历， postorder 是同一棵树的后序遍历，请你构造并返回这颗 二叉树 。
    """

    @staticmethod
    def build_tree(in_order: List[int], post_order: List[int]) -> TreeNode:
        def helper(in_left, in_right):
            # 如果这里没有节点构造二叉树了，就结束
            if in_left > in_right:
                return None

            # 选择 post_idx 位置的元素作为当前子树根节点
            val = post_order.pop()
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


class Solution47:
    """
    二叉树的层序遍历 II
    https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/
    给你二叉树的根节点 root ，返回其节点值 自底向上的层序遍历 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
    """

    @staticmethod
    def level_order_bottom(root: TreeNode) -> List[List[int]]:
        queue = []
        res = []
        if root:
            queue.append(root)
        while queue:
            q_size = len(queue)
            tmp = []
            for _ in range(q_size):
                r = queue.pop(0)
                tmp.append(r.val)
                if r.left:
                    queue.append(r.left)
                if r.right:
                    queue.append(r.right)
            res.append(tmp)
        return res[::-1]


class Solution48:
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


class Solution49:
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


class Solution50:
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

# -*- coding: utf-8 -*-
import sys
import heapq
import itertools
import collections
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
    假设数组 nums 的值全都严格大于 00，那么一定有 dp[i] = dp[i - 1] + nums[i]。
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
    我们用 f(i, j)f(i,j) 表示从左上角走到 (i, j)(i,j) 的路径数量，其中 ii 和 jj 的范围分别是 [0, m)[0,m) 和 [0, n)[0,n)。
    由于我们每一步只能从向下或者向右移动一步，因此要想走到 (i, j)(i,j)，如果向下走一步，那么会从 (i-1, j)(i−1,j) 走过来；
    如果向右走一步，那么会从 (i, j-1)(i,j−1) 走过来。
    """

    @staticmethod
    def unique_paths(m: int, n: int) -> int:
        f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]  # 第一行和第一例默认为1
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[m - 1][n - 1]


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
    对于不在第一行和第一列的元素，可以从其上方相邻元素向下移动一步到达，或者从其左方相邻元素向右移动一步到达，元素对应的最小路径和等于其上方相邻元素与其左方相邻元素两者对应的最小路径和中的最小值加上当前元素的值。由于每个元素对应的最小路径和与其相邻元素对应的最小路径和有关，因此可以使用动态规划求解。
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
        for i in range(n - 1, -1, -1):  # 对数组进行一次逆序遍历，找出第一个不为 99 的元素，将其加一并将后续所有元素置零即可
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
    给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

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
        D = [[0] * (m + 1) for _ in range(n + 1)]

        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j

        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)

        return D[n][m]


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
    """

    @staticmethod
    def min_window(s: str, t: str) -> str:
        need_dict = collections.defaultdict(int)  # 用来装分别需要的字符数量，包括必要的和非必要的，需要的必要字符数量>0，非必要的字符数量<0
        for ch in t:  # 初始化需要的必要字符数量
            need_dict[ch] += 1
        need_cnt = len(t)  # 用来判断总共需要多少字符才能达到要求
        i = j = 0
        res = [0, float('inf')]
        for j in range(len(s)):
            if need_dict[s[j]] > 0:  # 只有必要字符的数量才可能>0
                need_cnt -= 1
            need_dict[s[j]] -= 1  # 任意值都可以装进need_dict，但是非必要字符只可能<0
            if need_cnt == 0:  # 需要的字符都足够了
                while need_cnt == 0:  # 开始准备右移左指针，缩短距离
                    if j - i < res[1] - res[0]:  # 字符串更短，替换答案
                        res = [i, j]
                    need_dict[s[i]] += 1  # 在移动左指针之前先将左指针的值加回来，这里可以是非必要字符
                    if s[i] in t and need_dict[s[i]] > 0:  # 确认是必要字符且不多于所需要的数量（有多余的话只可能<=0，因为上一句我们已经将字符+1了）后，将need_cnt+1
                        need_cnt += 1
                    i += 1  # 右移左指针，寻找下一个符合的子串
        return s[res[0]:res[1] + 1] if res[1] - res[0] < len(s) else ''


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
    """

    @staticmethod
    def largest_rectangle_area(heights: List[int]) -> int:
        size = len(heights)
        res = 0
        heights = [0] + heights + [0]
        # 先放入哨兵结点，在循环中就不用做非空判断
        stack = [0]
        size += 2

        for i in range(1, size):
            while heights[i] < heights[stack[-1]]:
                cur_height = heights[stack.pop()]
                cur_width = i - stack[-1] - 1
                res = max(res, cur_height * cur_width)
            stack.append(i)
        return res




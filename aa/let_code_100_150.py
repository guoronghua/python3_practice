import sys
import heapq
import itertools
import functools
import collections
from collections import Counter
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution1:
    """
    二叉树的最小深度
    https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/
    给定一个二叉树，找出其最小深度.
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    """

    @staticmethod
    def min_depth(root: TreeNode) -> int:  # 广度优先
        if not root:
            return 0

        que = collections.deque([(root, 1)])
        while que:
            node, depth = que.popleft()
            if not node.left and not node.right:
                return depth
            if node.left:
                que.append((node.left, depth + 1))
            if node.right:
                que.append((node.right, depth + 1))

        return 0

    def min_depth_v2(self, root: TreeNode) -> int:  # 深度优先
        if not root:
            return 0

        if not root.left and not root.right:
            return 1

        min_depth = 10 ** 9
        if root.left:
            min_depth = min(self.min_depth_v2(root.left), min_depth)
        if root.right:
            min_depth = min(self.min_depth_v2(root.right), min_depth)

        return min_depth + 1


class Solution2:
    """
    路径总和
    https://leetcode-cn.com/problems/path-sum/
    给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，
    这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。
    """

    @staticmethod
    def has_path_sum(root: Optional[TreeNode], target: int) -> bool:  # 广度优先搜索
        if not root:
            return False
        queue = [(root, [root.val])]
        while queue:
            node, path = queue.pop(0)
            if not node.left and not node.right and sum(path) == target:
                return True
            if node.left:
                queue.append((node.left, path + [node.left.val]))
            if node.right:
                queue.append((node.right, path + [node.right.val]))
        return False

    def has_path_sum_v2(self, root: TreeNode, target: int) -> bool:  # 递归
        if not root:
            return False
        if not root.left and not root.right:
            return target == root.val
        return self.has_path_sum(root.left, target - root.val) or self.has_path_sum(root.right, target - root.val)


class Solution3:
    """
    路径总和 II
    https://leetcode-cn.com/problems/path-sum-ii/
    给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
    """

    @staticmethod
    def path_sum(root: Optional[TreeNode], target: int) -> List[List[int]]:  # 广度优先搜索
        res = []
        if not root:
            return []
        queue = [(root, [root.val])]
        while queue:
            node, path = queue.pop(0)
            if not node.left and not node.right and sum(path) == target:
                res.append(path)
            if node.left:
                queue.append((node.left, path + [node.left.val]))
            if node.right:
                queue.append((node.right, path + [node.right.val]))
        return res

    @staticmethod
    def path_sum(root: TreeNode, target: int) -> List[List[int]]:  # 深度优先搜索
        ret = list()
        path = list()

        def dfs(root: TreeNode, target: int):
            if not root:
                return
            path.append(root.val)
            if not root.left and not root.right and target == sum(path):
                ret.append(path[:])
            dfs(root.left, target)
            dfs(root.right, target)
            path.pop()

        dfs(root, target)
        return ret


class Solution4:
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
        aa = []

        def pre_order_traversal(root: TreeNode):
            if root:
                pre_order_list.append(root)
                aa.append(root.val)
                pre_order_traversal(root.left)
                pre_order_traversal(root.right)

        pre_order_traversal(root)
        size = len(pre_order_list)
        print(aa)
        for i in range(1, size):
            prev, curr = pre_order_list[i - 1], pre_order_list[i]
            prev.left = None
            prev.right = curr

    @staticmethod
    def flatten_v2(root: TreeNode) -> None:
        if not root:
            return

        stack = [root]
        prev = None

        while stack:
            curr = stack.pop()
            if prev:
                prev.left = None
                prev.right = curr
            left, right = curr.left, curr.right
            if right:
                stack.append(right)
            if left:
                stack.append(left)
            prev = curr


class Solution5:
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
        @functools.cache  # Python functools记忆化模块，等价于哈希表记录的功能
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

        dp = [[0] * (n + 1) for _ in range(
            m + 1)]  # 创建二维数组 dp[i][j] 表示在 s[i:] 的子序列中 t[j:] 出现的个数。s[i:] 表示 s 从下标 i 到末尾的子字符串，t[j:] 表示 t 从下标 j 到末尾的子字符串
        for i in range(m + 1):  # 当 j=n 时, t[j:] 为空字符串，由于空字符串是任何字符串的子序列，因此对任意 0≤ i ≤m，有 dp[i][n]=1
            dp[i][n] = 1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:  # 当 s[i]=t[j] 时，如果 s[i] 和 t[j] 匹配，则考虑 t[j+1:] 作为 s[i+1:] 的子序列，子序列数为 dp[i+1][j+1]，
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][
                        j]  # 如果 s[i] 不和 t[j] 匹配，则考虑 t[j:] 作为 s[i+1:] 的子序列，子序列数为 dp[i+1][j]。
                else:
                    dp[i][j] = dp[i + 1][
                        j]  # 当 s[i]!=t[j] 时，s[i] 不能和 t[j] 匹配，因此只考虑 t[j:] 作为 s[i+1:] 的子序列，子序列数为 dp[i+1][j]。

        return dp[0][0]


class Solution6:
    """
    填充每个节点的下一个右侧节点指针
    https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/
    给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。
    填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
    初始状态下，所有 next 指针都被设置为 NULL。
    输入：root = [1,2,3,4,5,6,7]
    输出：[1,#,2,3,#,4,5,6,7,#]
    解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
    """

    @staticmethod
    def connect(root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        queue = [root]
        while queue:
            size = len(queue)
            pre = None
            for _ in range(size):
                node = queue.pop(0)
                node.next = pre
                pre = node
                if node.right:
                    queue.append(node.right)
                if node.left:
                    queue.append(node.left)
        return root

    @staticmethod
    def connect_v2(root: 'Node') -> 'Node':
        if not root:
            return root
        # 从根节点开始
        leftmost = root
        while leftmost.left:
            # 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
            head = leftmost
            while head:
                # 相同父节点的
                head.left.next = head.right
                # 不是相同父节点的，但同一层的
                if head.next:
                    head.right.next = head.next.left
                # 指针向后移动
                head = head.next
            # 去下一层的最左的节点
            leftmost = leftmost.left
        return root


class Solution7:
    """
    杨辉三角
    https://leetcode-cn.com/problems/pascals-triangle/
    给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
    在「杨辉三角」中，每个数是它左上方和右上方的数的和。
    输入: numRows = 5
    输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
    """

    @staticmethod
    def generate(num_rows: int) -> List[List[int]]:
        res = [[1]]
        if num_rows == 1:
            return res
        for i in range(2, num_rows + 1):
            pre = res[-1]
            temp = [1]
            for j in range(1, len(pre)):
                temp.append(pre[j - 1] + pre[j])
            temp.append(1)
            res.append(temp)
        return res


class Solution8:
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

print([2,4,5,6,7].index(5))
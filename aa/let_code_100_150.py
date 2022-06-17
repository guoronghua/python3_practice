import sys
import heapq
import itertools
import functools
import collections
from collections import Counter
from typing import List, Optional
from collections import defaultdict
from typing import List
from collections import deque
import string


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
                # 指针向后移动，遍历完这一层
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


class Solution9:
    """
    根据身高重建队列
    https://leetcode.cn/problems/queue-reconstruction-by-height/
    假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
    请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。
    输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    解释：
    编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
    编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
    编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
    编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
    编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
    编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
    因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
    """

    @staticmethod
    def reconstruct_queue(people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))  # 将每个人按照身高从大到小进行排序,按照 hi 为第一关键字降序，ki为第二关键字升序进行排序
        ans = list()
        for person in people:
            ans[person[1]:person[1]] = [person]
        return ans


class ListNode(object):
    def __init__(self, val, nx=None):
        self.val = val
        self.next = nx


class Solution10:
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


class Node:
    def __init__(self, data, _pre=None, _next=None):
        self.data = data
        self.pre = _pre
        self.next = _next

    def __str__(self):
        return str(self.data)


class DoublyLink:
    """
    双向链表的实现
    """

    def __init__(self):
        self.tail = None
        self.head = None
        self.size = 0

    def __str__(self):
        str_text = ""
        cur_node = self.head
        while cur_node:
            str_text += cur_node.data + " "
            cur_node = cur_node.next
        return str_text

    def insert(self, data):
        if isinstance(data, Node):
            tmp_node = data
        else:
            tmp_node = Node(data)

        if self.size == 0:
            self.tail = tmp_node
            self.head = self.tail

        else:
            self.head._pre = tmp_node
            tmp_node._next = self.head
            self.head = tmp_node

        self.size += 1
        return tmp_node

    def remove(self, node):
        if node == self.head:
            self.head.next.pre = None
            self.head = self.head.next

        elif node == self.tail:
            self.tail.pre.next = None
            self.tail = self.tail.pre

        else:
            node.pre.next = node.next
            node.next.pre = node.pre
        self.size -= 1


class LRUCache:
    """
    LRU cache 算法实现
    插入数据时：若空间满了，则删除链表尾部元素，在进行插入
    查询数据时：先把数据删除，再重新插入数据，保证了元素的顺序是按照访问顺序排列
    """

    def __init__(self, size):
        self.size = size
        self.hash_map = dict()
        self.link = DoublyLink()

    def set(self, key, value):

        if self.size == self.link.size:
            self.link.remove(self.link.tail)

        if key in self.hash_map:
            self.link.remove(self.hash_map.get(key))

        tmp_node = self.link.insert(value)
        self.hash_map[key] = tmp_node

    def get(self, key):
        tmp_node = self.hash_map.get(key)
        self.link.remove(tmp_node)
        self.link.insert(tmp_node)
        return tmp_node.data


from collections import OrderedDict


class LRUCacheV2:
    """
    OrderedDict的本质就是一个有序的dict，其实现也是通过一个dict+双向链表
    """

    def __init__(self, size):
        self.size = size
        self.linked_map = OrderedDict()

    def set(self, key, value):
        if key in self.linked_map:
            self.linked_map.pop(key)

        if self.size == len(self.linked_map):
            self.linked_map.popitem(last=False)
        self.linked_map.update({key: value})

    def get(self, key):
        value = self.linked_map.get(key)
        self.linked_map.pop(key)
        self.linked_map.update({key: value})
        return value


import time
from functools import wraps


class Solution11:
    """
    重试装饰器的实现
    """

    @staticmethod
    def retry(tries=4, delay=3):
        def wrapper(f):
            @wraps(f)
            def f_retry(*args, **kwargs):
                m_tries, m_delay = tries, delay
                while m_tries > 1:
                    try:
                        return f(*args, **kwargs)
                    except Exception as _:
                        time.sleep(m_delay)
                        m_tries -= 1
                return f(*args, **kwargs)

            return f_retry  # true decorator

        return wrapper

    @staticmethod
    def log(func):
        # 普通装饰器的写法
        def wrapper(*args, **kw):
            print('call %s():' % func.__name__)
            return func(*args, **kw)

        return wrapper


class Solution12:
    """
    对链表进行插入排序
    https://leetcode.cn/problems/insertion-sort-list/
    给定单个链表的头 head ，使用 插入排序 对链表进行排序，并返回 排序后链表的头 。
    """

    @staticmethod
    def insertion_sort_list(head: ListNode) -> ListNode:
        if not head:
            return head

        dummy_head = ListNode(0)  # 创建哑节点 dummy_head
        dummy_head.next = head  # 令 dummy_head.next = head。引入哑节点是为了便于在 head 节点之前插入节点
        last_sorted = head  # 维护 last_sorted 为链表的已排序部分的最后一个节点，初始时 last_sorted = head。
        curr = head.next  # 维护 curr 为待插入的元素，初始时 curr = head.next。

        while curr:
            if last_sorted.val <= curr.val:  # 比较 last_sorted 和 curr 的节点值
                last_sorted = last_sorted.next  # 若 last_sorted.val <= curr.val，说明 curr 应该位于 last_sorted 之后，将 last_sorted 后移一位，curr 变成新的 last_sorted。
            else:
                prev = dummy_head  # 否则，从链表的头节点开始往后遍历链表中的节点，寻找插入 curr 的位置。令 prev 为插入 curr 的位置的前一个节点，进行如下操作，完成对 curr 的插入
                while prev.next.val <= curr.val:
                    prev = prev.next

                last_sorted.next = curr.next
                curr.next = prev.next
                prev.next = curr
            curr = last_sorted.next

        return dummy_head.next


class Solution13:
    """
    竖直打印单词
    https://leetcode.cn/problems/print-words-vertically/
    给你一个字符串 s。请你按照单词在 s 中的出现顺序将它们全部竖直返回。
    单词应该以字符串列表的形式返回，必要时用空格补位，但输出尾部的空格需要删除（不允许尾随空格）。
    每个单词只能放在一列上，每一列中也只能有一个单词。

    输入：s = "TO BE OR NOT TO BE"
    输出：["TBONTB","OEROOE","   T"]
    解释：题目允许使用空格补位，但不允许输出末尾出现空格。
    "TBONTB"
    "OEROOE"
    "   T"
    """

    @staticmethod
    def print_vertically(s: str) -> List[str]:
        words = s.split()
        max_len = max(len(word) for word in words)
        ans = list()
        for i in range(max_len):
            concat = "".join([word[i] if i < len(word) else " " for word in words])
            ans.append(concat.rstrip())
        return ans


class Solution14:
    """
    买卖股票的最佳时机
    https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/
    给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
    你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
    返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

    输入：[7,1,5,3,6,4]
    输出：5
    解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

    """

    @staticmethod
    def max_profit(prices: List[int]) -> int:
        inf = int(1e9)
        min_price = inf
        max_profit = 0
        for price in prices:
            max_profit = max(price - min_price, max_profit)
            min_price = min(min_price, price)
        return max_profit


class Solution15:
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
        inf = int(1e9)
        min_price = inf
        max_profit = 0
        for price in prices:
            max_profit = max(price - min_price, max_profit)
            min_price = min(min_price, price)
        return max_profit


class Solution16:
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


class Solution17:
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


class Solution18:
    """
    单词接龙 II
    https://leetcode.cn/problems/word-ladder-ii/
    按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的 转换序列 是形式上像 beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：

    每对相邻的单词之间仅有单个字母不同。
    转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
    sk == endWord
    给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。
    """

    def find_ladders(self, begin_word: str, end_word: str, word_list: List[str]) -> List[List[str]]:
        # 先将 word_list 放到哈希表里，便于判断某个单词是否在 word_list 里
        word_set = set(word_list)
        res = []
        if len(word_set) == 0 or end_word not in word_set:
            return res

        successors = defaultdict(set)
        # 第 1 步：使用广度优先遍历得到后继结点列表 successors
        # key：字符串，value：广度优先遍历过程中 key 的后继结点列表

        found = self.__bfs(begin_word, end_word, word_set, successors)
        if not found:
            return res
        # 第 2 步：基于后继结点列表 successors ，使用回溯算法得到所有最短路径列表
        path = [begin_word]
        self.__dfs(begin_word, end_word, successors, path, res)
        return res

    @staticmethod
    def __bfs(begin_word, end_word, word_set, successors):
        queue = deque()
        queue.append(begin_word)

        visited = set()
        visited.add(begin_word)

        found = False
        word_len = len(begin_word)
        next_level_visited = set()

        while queue:
            current_size = len(queue)
            for i in range(current_size):
                current_word = queue.popleft()
                word_list = list(current_word)

                for j in range(word_len):
                    origin_char = word_list[j]

                    for k in string.ascii_lowercase:
                        word_list[j] = k
                        next_word = ''.join(word_list)

                        if next_word in word_set:
                            if next_word not in visited:
                                if next_word == end_word:
                                    found = True

                                # 避免下层元素重复加入队列
                                if next_word not in next_level_visited:
                                    next_level_visited.add(next_word)
                                    queue.append(next_word)

                                successors[current_word].add(next_word)
                    word_list[j] = origin_char
            if found:
                break
            # 取两集合全部的元素（并集，等价于将 next_level_visited 里的所有元素添加到 visited 里）
            visited |= next_level_visited
            next_level_visited.clear()
        return found

    def __dfs(self, begin_word, end_word, successors, path, res):
        if begin_word == end_word:
            res.append(path[:])
            return

        if begin_word not in successors:
            return

        successor_words = successors[begin_word]
        for next_word in successor_words:
            path.append(next_word)
            self.__dfs(next_word, end_word, successors, path, res)
            path.pop()


class Solution19:
    """
    单词接龙
    https://leetcode.cn/problems/word-ladder/
    字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列 beginWord -> s1 -> s2 -> ... -> sk：

    每一对相邻的单词只差一个字母。
     对于 1 <= i <= k 时，每个 si 都在 wordList 中。注意， beginWord 不需要在 wordList 中。
    sk == endWord
    给你两个单词 beginWord 和 endWord 和一个字典 wordList ，返回 从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0 。

    输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    输出：5
    解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
    """

    @staticmethod
    def ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
        word_set = set(word_list)
        if len(word_set) == 0 or end_word not in word_set:
            return 0

        if begin_word in word_set:
            word_set.remove(begin_word)

        queue = deque()
        queue.append(begin_word)

        visited = set(begin_word)

        word_len = len(begin_word)
        step = 1
        while queue:
            current_size = len(queue)
            for i in range(current_size):
                word = queue.popleft()

                word_list = list(word)
                for j in range(word_len):
                    origin_char = word_list[j]

                    for k in range(26):
                        word_list[j] = chr(ord('a') + k)
                        next_word = ''.join(word_list)
                        if next_word in word_set:
                            if next_word == end_word:
                                return step + 1
                            if next_word not in visited:
                                queue.append(next_word)
                                visited.add(next_word)
                    word_list[j] = origin_char
            step += 1
        return 0

    # 双向广度优先遍历
    # 已知目标顶点的情况下，可以分别从起点和目标顶点（终点）执行广度优先遍历，直到遍历的部分有交集。这种方式搜索的单词数量会更小一些；
    # 更合理的做法是，每次从单词数量小的集合开始扩散；这里 beginVisited 和 endVisited 交替使用，等价于单向 BFS 里使用队列，
    # 每次扩散都要加到总的 visited 里。
    @staticmethod
    def ladder_length_v2(begin_word: str, end_word: str, word_list: List[str]) -> int:
        word_set = set(word_list)
        if len(word_set) == 0 or end_word not in word_set:
            return 0

        if begin_word in word_set:
            word_set.remove(begin_word)

        visited = set()
        visited.add(begin_word)
        visited.add(end_word)

        begin_visited = set()
        begin_visited.add(begin_word)

        end_visited = set()
        end_visited.add(end_word)

        word_len = len(begin_word)
        step = 1
        # 简化成 while begin_visited 亦可
        while begin_visited and end_visited:
            # 打开帮助调试
            # print(begin_visited)
            # print(end_visited)

            if len(begin_visited) > len(end_visited):
                begin_visited, end_visited = end_visited, begin_visited

            next_level_visited = set()
            for word in begin_visited:
                word_list = list(word)

                for j in range(word_len):
                    origin_char = word_list[j]
                    for k in range(26):
                        word_list[j] = chr(ord('a') + k)
                        next_word = ''.join(word_list)
                        if next_word in word_set:
                            if next_word in end_visited:
                                return step + 1
                            if next_word not in visited:
                                next_level_visited.add(next_word)
                                visited.add(next_word)
                    word_list[j] = origin_char
            begin_visited = next_level_visited
            step += 1
        return 0


class Solution20:
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


class Solution22:
    """
    求根节点到叶节点数字之和
    https://leetcode.cn/problems/sum-root-to-leaf-numbers/
    给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
    每条从根节点到叶节点的路径都代表一个数字
    例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
    计算从根节点到叶节点生成的 所有数字之和 。
    叶节点 是指没有子节点的节点。
    输入：root = [4,9,0,5,1]
    输出：1026
    解释：
    从根到叶子节点路径 4->9->5 代表数字 495
    从根到叶子节点路径 4->9->1 代表数字 491
    从根到叶子节点路径 4->0 代表数字 40
    因此，数字总和 = 495 + 491 + 40 = 1026

    """

    @staticmethod
    def sum_numbers_dfs(root: TreeNode) -> int:  # 深度优先搜索
        def dfs(_root: TreeNode, prev_total: int) -> int:
            if not _root:
                return 0
            total = prev_total * 10 + _root.val
            if not _root.left and not _root.right:
                return total
            else:
                return dfs(_root.left, total) + dfs(_root.right, total)

        return dfs(root, 0)

    @staticmethod
    def sum_numbers_bfs(root: TreeNode) -> int:  # 广度优先搜索
        if not root:
            return 0
        total = 0
        queue = [(root, root.val)]
        while queue:
            node, num = queue.pop(0)
            if not node.left and not node.right:
                total += num
            if node.left:
                queue.append((node.left, num * 10 + node.left.val))
            if node.right:
                queue.append((node.right, num * 10 + node.right.val))
        return total


class Solution23:
    """
    被围绕的区域
    给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
    输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
    """

    @staticmethod
    def solve_dfs(board: List[List[str]]) -> None:  # 深度优先搜索
        if not board:
            return

        n, m = len(board), len(board[0])

        def dfs(x, y):
            if not 0 <= x < n or not 0 <= y < m or board[x][y] != 'O':
                return

            board[x][y] = "A"  # 把标记过的字母 O 修改为字母 A。
            dfs(x + 1, y)
            dfs(x - 1, y)
            dfs(x, y + 1)
            dfs(x, y - 1)

        for i in range(n):  # 对于每一个边界上的 O，我们以它为起点，标记所有与它直接或间接相连的字母 O；
            dfs(i, 0)
            dfs(i, m - 1)

        for i in range(m):  # 对于每一个边界上的 O，我们以它为起点，标记所有与它直接或间接相连的字母 O；
            dfs(0, i)
            dfs(n - 1, i)

        for i in range(n):
            for j in range(m):
                if board[i][j] == "A":  # 如果是标记过的修改回 0
                    board[i][j] = "O"
                elif board[i][j] == "O":  # 如果是非标记过的，用 X 填充
                    board[i][j] = "X"

    @staticmethod
    def solve_bfs(board: List[List[str]]) -> None:  # 广度优先搜索
        if not board:
            return

        n, m = len(board), len(board[0])
        que = collections.deque()

        # 处理边界情况
        for i in range(n):
            if board[i][0] == "O":  # 如果是标记过的改为 A
                que.append((i, 0))
                board[i][0] = "A"
            if board[i][m - 1] == "O":
                que.append((i, m - 1))
                board[i][m - 1] = "A"
        for i in range(m):
            if board[0][i] == "O":  # 如果是标记过的改为 A
                que.append((0, i))
                board[0][i] = "A"
            if board[n - 1][i] == "O":
                que.append((n - 1, i))
                board[n - 1][i] = "A"

        # 处理与边界相邻的情况
        while que:
            x, y = que.popleft()
            for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= mx < n and 0 <= my < m and board[mx][my] == "O":  # 如果是标记过的改为 A
                    que.append((mx, my))
                    board[mx][my] = "A"

        for i in range(n):
            for j in range(m):
                if board[i][j] == "A":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"


class Solution24(object):
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


class Solution25:
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

    @staticmethod
    def min_cut_v2(s: str):
        n = len(s)
        g = [[True] * n for _ in range(n)]

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                g[i][j] = (s[i] == s[j]) and g[i + 1][j - 1]

        f = [float("inf")] * n
        for i in range(n):
            if g[0][i]:
                f[i] = 0
            else:
                for j in range(i):
                    if g[j + 1][i]:
                        f[i] = min(f[i], f[j] + 1)
        return f[n - 1]


class Node(object):
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution26(object):
    """
    克隆图
    https://leetcode.cn/problems/clone-graph/
    给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。
    图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

    """

    def __init__(self):
        self.visited = {}

    def clone_graph_dfs(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node

        # 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if node in self.visited:
            return self.visited[node]

        # 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
        clone_node = Node(node.val, [])

        # 哈希表存储
        self.visited[node] = clone_node

        # 遍历该节点的邻居并更新克隆节点的邻居列表
        if node.neighbors:
            clone_node.neighbors = [self.clone_graph_dfs(n) for n in node.neighbors]

        return clone_node

    @staticmethod
    def clone_graph_bfs(node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node

        visited = {}

        # 将题目给定的节点添加到队列
        queue = deque([node])
        # 克隆第一个节点并存储到哈希表中
        visited[node] = Node(node.val, [])

        # 广度优先搜索
        while queue:
            # 取出队列的头节点
            n = queue.popleft()
            # 遍历该节点的邻居
            for neighbor in n.neighbors:
                if neighbor not in visited:
                    # 如果没有被访问过，就克隆并存储在哈希表中
                    visited[neighbor] = Node(neighbor.val, [])
                    # 将邻居节点加入队列中
                    queue.append(neighbor)
                # 更新当前节点的邻居列表
                visited[n].neighbors.append(visited[neighbor])
        return visited[node]


class Solution27:
    """
    加油站
    https://leetcode.cn/problems/gas-station/
    在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
    你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
    给定两个整数数组 gas 和 cost ，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的
    """
    @staticmethod
    def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1

        '''sum(gas) >= sum(cost)，一定有解【题目保证唯一解】'''
        n = len(gas)
        start = 0  # 记录出发点，从索引0开始
        total = 0  # 记录汽车实际油量
        for i in range(n):
            total += gas[i] - cost[i]  # 每个站点加油量相当于 gas[i] - cost[i]
            if total < 0:  # 在i处的油量<0，说明从之前站点出发的车均无法到达i
                start = i + 1  # 尝试从下一个站点i+1重新出发
                total = 0  # 重新出发时油量置为0

        return start  # 解是唯一的


class Solution28:
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

from functools import reduce
class Solution29:
    """
    只出现一次的数字
    https://leetcode.cn/problems/single-number/
    给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
    任何数和 0 做异或运算，结果仍然是原来的数
    任何数和其自身做异或运算，结果是 0
    """
    @staticmethod
    def single_number(nums: List[int]) -> int:
        return reduce(lambda x, y: x ^ y, nums)


class Solution30(object):
    """
    复制带随机指针的链表
    https://leetcode.cn/problems/copy-list-with-random-pointer/
    给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
    构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random
    指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
    """
    def __init__(self):
        self.cached_node = {}

    def copy_random_list(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None
        if not self.cached_node.get(head):
            head_new = Node(head.val)
            self.cached_node[head] = head_new
            head_new.next = self.copy_random_list(head.next)
            head_new.random = self.copy_random_list(head.random)
        return self.cached_node.get(head)


class Solution31:
    """
    单词拆分
    给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
    不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
    https://leetcode.cn/problems/word-break/
    """
    @staticmethod
    def word_break(s: str, word_dict: List[str]) -> bool:
        n=len(s)
        dp=[False]*(n+1)  # dp[i] 表示 s 的前 i 位是否可以用 wordDict 中的单词表示。
        dp[0]=True    # 初始化 dp[0]=True，空字符可以被表示。
        for i in range(n):  # 遍历字符串的所有子串
            for j in range(i+1,n+1):
                if dp[i] and (s[i:j] in word_dict): # dp[i]=True 说明 s 的前 i 位可以用 wordDict 表示, s[i:j]出现在 wordDict 中，说明 s 的前 j 位可以表示。
                    dp[j]=True
        return dp[-1]


class Solution32:
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


class Solution33(object):
    """
    环形链表 II
    https://leetcode.cn/problems/linked-list-cycle-ii/
    给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    """
    @staticmethod
    def detect_cycle(head):
        fast, slow = head, head
        while True:
            if not (fast and fast.next):
                return
            fast, slow = fast.next.next, slow.next
            if fast == slow:
                break
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast

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
        @functools.lru_cache # Python functools 记忆化模块，等价于哈希表记录的功能
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


import functools
import hashlib
import collections
from collections import Counter
from typing import List, Optional
from collections import defaultdict
from typing import List
from collections import deque
import string


class Solution1:
    """
    计数质数
    https://leetcode.cn/problems/count-primes/
    给定整数 n ，返回 所有小于非负整数 n 的质数的数量 。
    埃式筛，把不大于根号 n 的所有质数的倍数剔除
    注意每次找当前素数 x 的倍数时，是从 x^2 开始的。因为如果 x > 2，
    那么 2*x 肯定被素数 2 给过滤了，最小未被过滤的肯定是 x^2
    """
    @staticmethod
    def count_primes(n: int) -> int:
        if n < 2:
            return 0
        dp = [1] * n
        dp[0]=dp[1] = 0

        for i in range(2, int(n**0.5)+1): # 从 2 开始将当前数字的倍数全都标记为合数。标记到 根号n时停止即可
            if dp[i] == 1:
                j = i*i
                while j < n:
                    dp[j] = 0
                    j+=i
        return sum(dp)


class Solution2:
    """
    课程表
    https://leetcode.cn/problems/course-schedule/
    一共有 n 门课要上，编号为 0 ~ n-1
    先决条件[1, 0]，意思是必须先上课 0，才能上课 1。
    给你 n 、和一个先决条件表，请你判断能否完成所有课程。
    示例：n = 6，先决条件表：[[3, 0], [3, 1], [4, 1], [4, 2], [5, 3], [5, 4]]
    课 0, 1, 2 没有先修课，可以直接选。其余的课，都有两门先修课。
    我们用有向图来展现这种依赖关系（做事情的先后关系）
    如果存在一条有向边 A --> B，则这条边给 A 增加了 1 个出度，给 B 增加了 1 个入度。
    所以，顶点 0、1、2 的入度为 0。顶点 3、4、5 的入度为 2。
    0 -> 3 -> 5
    1 -> 3 -> 5
    1 -> 4 -> 5
    2 -> 4 -> 5
    """
    @staticmethod
    def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
        edges = collections.defaultdict(list)
        in_deg = [0] * num_courses

        for info in prerequisites:
            edges[info[1]].append(info[0])
            in_deg[info[0]] += 1 # 统计每个节点的入度

        # 让入度为 0 的课入列，它们是能直接选的课。
        q = collections.deque([u for u in range(num_courses) if in_deg[u] == 0])
        visited = 0

        while q:
            visited += 1
            u = q.popleft()  # 然后逐个出列，出列代表着课被选，需要减小相关课的入度。
            for v in edges[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0: # 如果相关课的入度新变为 0，安排它入列、再出列……直到没有入度为 0 的课可入列。
                    q.append(v)

        return visited == num_courses


class Solution3:
    """
    长度最小的子数组
    https://leetcode.cn/problems/minimum-size-subarray-sum/
    给定一个含有 n 个正整数的数组和一个正整数 target 。
    找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
    """
    @staticmethod
    def min_sub_array_len(target: int, nums: List[int]) -> int:
        if not nums:
            return 0

        n = len(nums)
        ans = n + 1
        start, end = 0, 0
        total = 0
        while end < n:
            total += nums[end]            # 滑动窗口
            while total >= target:
                ans = min(ans, end - start + 1)
                total -= nums[start]
                start += 1
            end += 1
        return 0 if ans == n + 1 else ans


class Solution4:
    """
    添加与搜索单词 - 数据结构设计
    https://leetcode.cn/problems/design-add-and-search-words-data-structure/
    请你设计一个数据结构，支持 添加新单词 和 查找字符串是否与任何先前添加的字符串匹配 。
    WordDictionary() 初始化词典对象
    void addWord(word) 将 word 添加到数据结构中，之后可以对它进行匹配
    bool search(word) 如果数据结构中存在字符串 word 匹配，则返回 true ；否则，返回 false 。word 中可能包含一些 '.' ，每个. 都可以表示任何一个字母。

    """
    def __init__(self):
        self.root = {}
        self.root = collections.defaultdict(set)

    def add_word(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur['#'] = True


    def dfs(self, node, word, i):
        if i == len(word):
            return '#' in node
        if word[i] == '.':
            for k, v in node.items():
                if k != '#' and self.dfs(v, word, i + 1):
                    return True
            return False
        else:
            if word[i] not in node:
                return False
            return self.dfs(node[word[i]], word, i + 1)

    def search(self, word: str) -> bool:
        return self.dfs(self.root, word, 0)

    def add_word_v2(self, word: str) -> None:
        self.root[len(word)].add(word)

    def search_v2(self, word: str) -> bool:
        if '.' not in word:
            return word in self.root[len(word)]
        else:
            for w in self.root[len(word)]:
                for i in range(len(word)):
                    if word[i] == '.':
                        continue
                    elif word[i] == w[i]:
                        continue
                    else:
                        break
                else:
                    return True
            return False


class Solution5:
    """
    单词搜索 II
    https://leetcode.cn/problems/word-search-ii/
    给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words， 返回所有二维网格上的单词 。
    单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

    """
    @staticmethod
    def find_words(board: List[List[str]], words: List[str]) -> List[str]:
        # 建立前缀树
        _trie = {}
        for word in words:
            node = _trie  # 头节点
            for w in word:
                if w not in node:
                    node[w] = {}
                node = node[w]
            node['end'] = 1

        res = []

        def dfs(r, c, trie, s):
            letter = board[r][c]
            # 定位到有letter的那颗树
            trie = trie[letter]
            if 'end' in trie and trie['end'] == 1:
                res.append(s + letter)
                trie['end'] = 0  # 标志位，遍历到该路径终止时剪枝
            board[r][c] = '#'  # 将遍历过的单元格标为#，确保在同一个单词中不被重复使用
            for x, y in ([r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]):
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] in trie:
                    dfs(x, y, trie, s + letter)
            board[r][c] = letter  # 该单词结束，将board复原

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in _trie:
                    dfs(i, j, _trie, '')
        return res





class Solution6:
    """
    最短回文串
    https://leetcode.cn/problems/shortest-palindrome/
    给定一个字符串 s，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。
    """

    @staticmethod
    def shortest_palindrome(s: str) -> str:
        def is_hw(_s):
            md5hash_s1 = hashlib.md5(_s.encode("utf-8"))
            md5hash_s2 = hashlib.md5(_s[::-1].encode("utf-8"))
            md5_s1 = md5hash_s1.hexdigest()
            md5_s2 = md5hash_s2.hexdigest()
            return md5_s1 == md5_s2

        if not s or len(s) == 1 or is_hw(s):
            return s

        max_hw = s[0]  # 在原字符串中找最长回文子串
        the_other = s[1:]
        for i in range(len(s) - 1, -1, -1):
            if is_hw(s[:i]):
                max_hw = s[:i]
                the_other = s[len(max_hw):]
                break
        return the_other[::-1] + max_hw + the_other
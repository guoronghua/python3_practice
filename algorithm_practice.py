# -*- coding: utf-8 -*-
# import re
import functools
import sys
from collections import deque



class Sort(object):
    """常见的排序算法"""

    def __init__(self):
        pass

    @staticmethod
    def bubble_sort(a_list):
        """冒泡排序"""
        list_length = len(a_list)
        if list_length <= 1:
            return a_list

        list_length = len(a_list)
        for x in range(0, list_length):
            flag = False
            for y in range(0, list_length - x - 1):
                if a_list[y] > a_list[y + 1]:
                    temp = a_list[y]
                    a_list[y] = a_list[y + 1]
                    a_list[y + 1] = temp
                    flag = True
            if not flag:
                return a_list

    @staticmethod
    def insert_sort(a_list):
        """插入排序"""
        list_length = len(a_list)
        if list_length <= 1:
            return a_list

        for x in range(1, list_length):
            to_insert_value = a_list[x]
            y = x - 1

            while y >= 0:
                if to_insert_value < a_list[y]:
                    a_list[y + 1] = a_list[y]
                else:
                    break
                y -= 1
            a_list[y + 1] = to_insert_value
        return a_list

    @staticmethod
    def select_sort(a_list):
        """选择排序"""
        list_length = len(a_list)
        if list_length <= 1:
            return a_list
        for x in range(0, list_length):
            min_index = x
            for y in range(x + 1, list_length):
                if a_list[y] < a_list[min_index]:
                    min_index = y

            if min_index != x:
                a_list[x], a_list[min_index] = a_list[min_index], a_list[x]
        return a_list

    def merge_sort(self, a_list):
        """归并排序"""
        if len(a_list) <= 1:
            return a_list

        mid = len(a_list) / 2
        left = self.merge_sort(a_list[:mid])
        right = self.merge_sort(a_list[mid:])

        return self._merge(left, right)

    @staticmethod
    def _merge(left, right):
        """归并排序之合并"""
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:]
        result += right[j:]
        return result

    def easy_quick_sort(self, a_list):
        """快速排序简单实现"""
        if len(a_list) < 2:
            return a_list

        pivot = a_list[0]

        less_than_pivot = [x for x in a_list if x < pivot]
        more_than_pivot = [x for x in a_list if x > pivot]
        return self.easy_quick_sort(less_than_pivot) + [pivot] + self.easy_quick_sort(more_than_pivot)

    def quick_sort(self, a_list):
        """快速排序"""
        return self._quick_sort(a_list, 0, len(a_list) - 1)

    def _quick_sort(self, a_list, left, right):
        if left < right:
            pivot = self._partition(a_list, left, right)
            self._quick_sort(a_list, left, pivot - 1)
            self._quick_sort(a_list, pivot + 1, right)
        return a_list

    @staticmethod
    def _partition(a_list, left, right):
        pivot_key = a_list[left]
        while left < right:
            while left < right and a_list[right] >= pivot_key:
                right -= 1
            a_list[left] = a_list[right]

            while left < right and a_list[left] <= pivot_key:
                left += 1
            a_list[right] = a_list[left]
        a_list[left] = pivot_key
        return left

    @staticmethod
    def bucket_sort(a_list):
        """桶排序，支持负数和2位小数"""
        if not a_list or len(a_list) < 2:
            return a_list

        accuracy = 1
        _max = int(max(a_list) * accuracy)
        _min = int(min(a_list) * accuracy)
        buckets = [0 for _ in range(_min, _max + 1)]

        for x in a_list:
            buckets[int(x) * accuracy - _min] += 1

        current = _min
        n = 0
        for bucket in buckets:
            while bucket > 0:
                a_list[n] = current / accuracy
                bucket -= 1
                n += 1
            current += 1
        return a_list

    @staticmethod
    def count_sort(a_list):
        """计数排序, 不支持负数"""
        if not a_list or len(a_list) < 2:
            return a_list

        _max = max(a_list)
        buckets = [0 for _ in xrange(0, _max + 1)]
        for x in a_list:
            buckets[x] += 1

        for x in range(1, _max + 1):
            buckets[x] += buckets[x - 1]
        result = [0 for _ in range(0, len(a_list))]

        for x in range(0, len(a_list)):
            index = buckets[a_list[x]] - 1
            result[index] = a_list[x]
            buckets[a_list[x]] -= 1
        return result

    @staticmethod
    def radix_sort(a_list):
        """基数排序"""
        i = 0
        _max = max(a_list)
        j = len(str(_max))  # 记录最大值的位数
        while i < j:
            bucket_list = [[] for _ in range(10)]
            for x in a_list:
                bucket_list[int(x / (10 ** i)) % 10].append(x)
            a_list = [y for x in bucket_list for y in x]
            i += 1
        return a_list


class Find(object):
    """常见的查找算法"""

    def __init__(self):
        pass

    @staticmethod
    def b_search(a_list, target):
        """二分查找算法 非递归实现"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target == a_list[mid]:
                return mid
            if target < a_list[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def b_recursion_search(self, a_list, target, low, high):
        """二分查找算法 非递归实现"""
        if low > high:
            return -1
        mid = low + (high - low) / 2
        if target == a_list[mid]:
            return mid
        if target < a_list[mid]:
            return self.b_recursion_search(a_list, target, low, mid - 1)
        else:
            return self.b_recursion_search(a_list, target, mid + 1, high)

    @staticmethod
    def b_first_search(a_list, target):
        """查找第一个等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target <= a_list[mid]:
                if target == a_list[mid] and target != a_list[mid - 1]:
                    return mid
                else:
                    high = mid - 1
            else:
                low = mid + 1
        return -1

    @staticmethod
    def b_last_search(a_list, target):
        """查找最后一个等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target >= a_list[mid]:
                if target == a_list[mid] and target != a_list[mid + 1]:
                    return mid
                else:
                    low = mid + 1
            else:
                high = mid - 1
        return -1

    @staticmethod
    def b_last_than_search(a_list, target):
        """查找最后一个大于等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target <= a_list[mid]:
                if mid == 0 or target > a_list[mid - 1]:
                    return mid
                else:
                    high = mid - 1
            else:
                low = mid + 1
        return -1

    @staticmethod
    def b_last_less_search(a_list, target):
        """查找最后一个小于等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target >= a_list[mid]:
                if mid == len(a_list) - 1 or target < a_list[mid + 1]:
                    return mid
                else:
                    low = mid + 1
            else:
                high = mid - 1
        return -1

    @staticmethod
    def sqrt(x):
        """求一个数的算数平方根，精确到小数点后6位"""
        low = 0
        mid = x / 2.0
        high = x
        while abs(mid ** 2 - x) > 0.000001:
            if mid ** 2 < x:
                low = mid
            else:
                high = mid
            mid = (low + high) / 2
        return round(mid, 6)


def generate_bc(sub_str):
    """
    :param sub_str: 是模式串
    :return: 返回一个散列表  散列表的下标是 b_char 中每个字符的 ascii 码值，下标对应的值是该字符在b_char中最后一次出现的位置
    """
    ascii_size = 256
    ascii_list = [-1] * ascii_size  # 初始化一个散列表

    for i in range(0, len(sub_str)):
        ascii = ord(sub_str[i])  # 计算 b_char中每个字符的 ascii 值
        ascii_list[ascii] = i  # 存每个字符在 b_char 最后一次出现的位置
    return ascii_list


def bm_simple(main_str, sub_str):
    """
    :param main_str: 主字符串
    :param sub_str: 模式串
    :return:
    """
    bc = generate_bc(sub_str)  # 记录模式串中每个字符最后出现的位置，也就是坏字符的哈希表
    i = 0  # 表示主串和模式串对齐的第一个字符
    n = len(main_str)  # 表示主串的长度
    m = len(sub_str)  # 表示子串的长度
    while i <= n - m:
        j = m - 1
        while j >= 0:
            if main_str[i + j] != sub_str[j]:  # 坏字符对应模式串中的下标是 j
                break
            j -= 1
        if j < 0:
            return i  # 匹配成功，返回主串和模式串第一个匹配的字符的位置
        bad_str_index = bc[ord(main_str[i + j])]  # 坏字符在模式串中的位置
        i += (j - bad_str_index)  # 这里等同于将模式串往后滑动  j - bc[ord(main_str[i + j]) 位
    return -1


def generate_gs(sub_str):
    m = len(sub_str)
    suffix = [-1] * m
    prefix = [False] * m
    for i in range(0, m - 1):
        j = i
        k = 0  # 公共后缀串长度，
        while j >= 0 and sub_str[j] == sub_str[m - 1 - k]:  # 求公共后缀子串，先计算第一个跟模式子串最后一个字符相匹配的位置
            j -= 1  # 然后依次比较前面的字符是否相等，比如：cabcab，先从前往后历遍，计算得到 b 字符是符合要求的，
            k += 1  # 此时再从 b 字符前一个位置与模式串倒数第二个位置的字符去比较，如果还是相等，则继续，循环了 k 次说明匹配的字符长度就是 k
            suffix[k] = j + 1  # j+1 表示公共后缀子串在 b[0, i]中的起始下标

        if j == -1:  # 如果公共后缀子串也是模式串的前缀子串
            prefix[k] = True
    return suffix, prefix


def move_by_gs(j, m, suffix, prefix):
    """j 表示坏字符对应的模式串中的字符下标，m 表示模式串的长度"""
    k = m - 1 - j  # k 好后缀长度
    if suffix[k] != -1:
        return j - suffix[k] + 1
    for r in range(j + 2, m):
        if prefix[m - r]:
            return r
    return m  # 如果前面两个规则都不使用，则直接往后滑动 m 位


def bm(main_str, sub_str):
    """
    :param main_str: 主字符串
    :param sub_str: 模式串
    :return:
    """
    bc = generate_bc(sub_str)  # 记录模式串中每个字符最后出现的位置，也就是坏字符的哈希表
    suffix, prefix = generate_gs(sub_str)

    i = 0  # 表示主串和模式串对齐的第一个字符
    n = len(main_str)  # 表示主串的长度
    m = len(sub_str)  # 表示子串的长度
    while i <= n - m:
        j = m - 1
        while j >= 0:
            if main_str[i + j] != sub_str[j]:  # 坏字符对应模式串中的下标是 j
                break
            j -= 1
        if j < 0:
            return i  # 匹配成功，返回主串和模式串第一个匹配的字符的位置

        x = j - bc[ord(main_str[i + j])]  # 计算坏字符规则每次需要往后滑动的次数
        y = 0
        if j < m - 1:  # 如果有好后缀的话
            y = move_by_gs(j, m, suffix, prefix)  # 计算好后缀规则每次需要往后滑动的次数
        i = i + max(x, y)

    return -1


def get_next_list(sub_str):
    """计算如： abababzabababa 每个位置的前缀集合与后缀集合的交集中最长元素的长度
       输出为： [-1, -1, 0, 1, 2, 3, -1, 0, 1, 2, 3, 4, 5, 4]
       计算最后一个字符 a 的逻辑如下：
       abababzababab 前缀后缀最大匹配了 6 个（ababab），次大匹配是4 个（abab），次大匹配的前缀后缀只可能在 ababab 中，
       所以次大匹配数就是 ababab 的最大匹配数，在数组中查到出该值为 3。第三大的匹配数同理，它既然比 3 要小，那前缀后缀也只能在 abab 中找，
       即 abab 的最大匹配数，查表可得该值为 1。再往下就没有更短的匹配了。来计算最后位置 a 的值：既然末尾字母不是 z，
       那么就不能直接 5+1=7 了，我们回退到次大匹配 abab，刚好 abab 之后的 a 与末尾的 a 匹配，所以 a 处的最大匹配数为 4。
    """
    m = len(sub_str)
    next_list = [None] * m
    next_list[0] = -1
    k = -1
    i = 1
    while i < m:
        while k != -1 and sub_str[k + 1] != sub_str[i]:  # k+1 是每个前缀子串的最后一个字符，i 是每个后缀子串的第一个字符
            k = next_list[k]  # 求次大匹配数，次大匹配数如 abab 是上层 ababab 的最大匹配数
        if sub_str[k + 1] == sub_str[i]:  # 如果前缀子串的最后一个字符和后缀子串的第一个字符相等，则增加前缀子串和后缀子串的长度再比较
            k += 1
        next_list[i] = k  # sub_str[0:i] 的前缀集合与后缀集合的交集中最长元素的长度
        i += 1
    return next_list


def kmp(main_str, sub_str):
    n = len(main_str)
    m = len(sub_str)
    next_list = get_next_list(sub_str)
    i = 0
    j = 0
    while i < n and j < m:
        if j == -1 or main_str[i] == sub_str[j]:  # 如果子串的每个字符和主串的每个字符相等就继续循环
            i += 1
            j += 1
        else:  # 在模式串j处失配，需要找到sub_str[0:j-1]的前缀集合与后缀集合的交集中最长元素的长度, 然后直接将j更新
            j = next_list[j]  # 为next_list[j]，本来是需要将main_str[i] 处的字符依次与 sub_str[0:j]中的每个元素依次再比较的
    if j == m:
        return i - j
    return -1


class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.Trie = {}

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curr = self.Trie
        for w in word:
            if w not in curr:
                curr[w] = {}
            curr = curr[w]
        curr['#'] = 1
        print(self.Trie)

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self.Trie
        for w in word:
            if w not in curr:
                return False
            curr = curr[w]
        return "#" in curr

    def starts_with(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curr = self.Trie
        for w in prefix:
            if w not in curr:
                return False
            curr = curr[w]
        return True


class TrieNode(object):
    __slots__ = ['value', 'next_node', 'fail', 'emit']  # 限制实例的属性只能是这四个

    def __init__(self, value):
        self.value = value
        self.next_node = dict()
        self.fail = None
        self.emit = None  # 当到达叶子节点时记录此时的路径，也就是一个完整的字符串


class AcAutoTrie(object):
    __slots__ = ['_root']  # 限制实例的属性只有一个

    def __init__(self, words):
        self._root = self._build_trie(words)
        self._build_fail()

    @staticmethod
    def _build_trie(words):
        """使用 words 构建 trie 树"""
        assert isinstance(words, list) and words
        root = TrieNode('root')
        for word in words:
            node = root
            for c in word:
                if c not in node.next_node:  # 如果字符在子节点中不存在，则以字符 c 为value 生成一个 TrieNode
                    node.next_node[c] = TrieNode(c)  # 并以字符 c 为 key 放入当前 node 的子节点中
                node = node.next_node.get(c)  # 并将当前的 node 更新为 以 c 为value 生成一个 TrieNode
            node.emit = word  # 历遍完一个 word 之后，说明到底树的叶子节点了

        return root

    def _build_fail(self):
        """构建失败指针"""
        queue = []
        queue.insert(0, (self._root, None))  # 将 root 节点和 root 节点的父节点放入到队列中
        while len(queue) > 0:
            node_parent = queue.pop()
            curr, parent = node_parent[0], node_parent[1]  # 取出队列中的一个 node 和其父 node
            for sub in curr.next_node.itervalues():  # 历遍当前 node  所有子 node 全部放入队列中
                queue.insert(0, (sub, curr))
            if parent is None:  # 如果当前节点是 root 节点则跳过循环
                continue
            elif parent is self._root:  # 如果当前节点的父节点是 root 则将当前节点的失败指针指向 root
                curr.fail = self._root
            else:
                fail = parent.fail  # 否则就一直找啊找, 这里的 fail 其实很多时候等于 root 的
                while fail and curr.value not in fail.next_node:
                    fail = fail.fail
                if fail:  # 直到有一个失败指针指向的节点假设叫X吧，当前节点的值在X的子节点里存在
                    curr.fail = fail.next_node.get(curr.value)  # 那么就把当前节点的失败指针指向X的那个与当前节点相等的子节点
                else:
                    curr.fail = self._root  # 如果全部都都找完了也没找到，那就把当前节点的失败指针指向 root

    def search(self, s):
        """AC 自动的搜索匹配"""
        seq_list = []  # 用来保存返回结果的列表
        node = self._root  # 从 root 节点开始
        for i, c in enumerate(s):  # 历遍字符串 s 里的每个字符，其中 i 是字符 c 在 s 里的索引
            while c not in node.next_node and node != self._root:  # 一直循环去查找，直到字符 c 在当前节点的子节点中存在，
                node = node.fail  # 或者当前节点是 root 节点了就停止查找
            node = node.next_node.get(c)  # 将当前的 node 更新为等于它自己的值等于 c 的那个子 node
            if not node:
                node = self._root  # 如果不存在这么一个子 node 那上面的 while 循环肯定是满足 node = self._root 的
            temp = node
            while temp != self._root:  # 如果此时字符 c 不匹配任何当前 node 的子 node 则跳过下面的循环，继续去找下一个字符
                if temp.emit:  # 否则，如果循环到字符 c 时候已经到了树的叶子节点了，说明找到了匹配的字符
                    from_index = i - len(temp.emit) + 1  # 计算匹配到的字符的起始位置
                    match_info = (from_index, temp.emit)
                    seq_list.append(match_info)
                temp = temp.fail
        return seq_list


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


class Backtracking(object):
    """回朔算法案列分析"""

    def __init__(self):
        self.result = [None] * 8  # 存储 8 皇后问题的结果，下标表示行，值表示皇后存储的列
        self.count = 0  # 八皇后所有可能排列位置数

    def cal_8_queens(self, row):
        if row == 8:  # 8个棋子都放置好了，打印结果
            self.count += 1
            self.print_queens()
            print('-' * 20)
            return

        for column in range(0, 8):  # 每一行有 8个放法
            if self.is_column_ok(row, column):
                self.result[row] = column  # 第 row 行的棋子放在 column 列
                self.cal_8_queens(row + 1)  # 计算下一个

    def is_column_ok(self, row, column):
        left_column = column - 1
        right_column = column + 1
        for i in range(row - 1, -1, -1):  # 逐行往上考察
            if self.result[i] in [column, left_column, right_column]:  # 检查上一行,对角线上, 右上对角线上是否有皇后
                return False
            left_column -= 1
            right_column -= 1
            i -= 1
        return True

    def print_queens(self):
        for row in range(0, 8):
            column_str = ""
            for column in range(0, 8):
                if self.result[row] == column:
                    column_str += "Q "
                else:
                    column_str += "* "
            print(column_str)


class Patten:
    """回朔法实现正则表达式"""

    def __init__(self, patten):
        self.matched = True
        self.patten = patten  # 正则表达式
        self.p_len = len(patten)  # 正则表达式长度

    def match(self, text):
        self.matched = False
        self.re_match(0, 0, text)
        return self.matched

    def re_match(self, text_index, patten_index, text):
        if self.matched:
            return
        if patten_index == self.p_len:  # 正在表达式到结尾了
            if text_index == len(text):  # 文本串也到结尾了
                self.matched = True
            return
        if self.patten[patten_index] == "*":  # 匹配任意字符
            for k in range(0, len(text) - text_index + 1):
                self.re_match(text_index + k, patten_index + 1, text)
        elif self.patten[patten_index] == "?":  # 0 个或一个字符
            self.re_match(text_index, patten_index + 1, text)
            self.re_match(text_index + 1, patten_index + 1, text)

        elif text_index < len(text) and self.patten[patten_index] == text[text_index]:  # 纯字符匹配才行
            self.re_match(text_index + 1, patten_index + 1, text)


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

    @staticmethod
    def backpack4():
        """有依赖情况的背包问题"""
        N, M = 3200, 60
        f = [0] * N
        # 分组背包，每组有四种情况，a.主件 b.主件+附件1 c.主件+附件2 d.主件+附件1+附件2
        v = [[0 for i in range(4)] for j in range(M)]  # 金额
        w = [[0 for i in range(4)] for j in range(M)]  # 价值

        # n, m = map(int, input().split())
        n, m = 1500, 7
        # n //= 10  # 价格为10的整数倍，节省时间

        for i in range(1, m + 1):
            for inp in ["500 1 0", "400 4 0", "300 5 1", "400 5 1", "200 5 0", "500 4 0", "400 4 0"]:
                x, y, z = map(int, inp.split())
                # x //= 10
                if z == 0:
                    for t in range(4):
                        v[i][t] = v[i][t] + x
                        w[i][t] = w[i][t] + x * y

                elif v[z][1] == v[z][0]:  # 如果a==b，添加附件1(如果a=b=c=d说明没有附件)
                    v[z][1], w[z][1] = v[z][1] + x, w[z][1] + x * y
                    v[z][3], w[z][3] = v[z][3] + x, w[z][3] + x * y

                else:  # 添加附件2
                    v[z][2], w[z][2] = v[z][2] + x, w[z][2] + x * y
                    v[z][3], w[z][3] = v[z][3] + x, w[z][3] + x * y

        for i in range(1, m + 1):
            for j in range(n, -1, -1):
                for k in range(4):
                    if j >= v[i][k]:
                        f[j] = max(f[j], f[j - v[i][k]] + w[i][k])
        print(9 * f[n])


def knapsack(items, n, w):
    """
    :param items: 物品重量列表如 [2,2,4,6,3]
    :param n: 物品个数
    :param w: 背包可承载的重量
    :return:
    """
    states = [[False for _ in range(w + 1)] for _ in range(n)]  # 默认值为 False
    states[0][0] = True  # 第一行的值要特殊处理，可以利用哨兵优化
    if items[0] <= w:
        states[0][items[0]] = True  # 开始第一次做选择

    for i in range(1, n):  # 动态规划状态转移
        for j in range(0, w + 1):  # 不把第 i 个物品放入背包
            if states[i - 1][j]:
                states[i][j] = True

        for j in range(0, w - items[i] + 1):  # 把第 i 个物品放入背包
            if states[i - 1][j]:
                states[i][j + items[i]] = True

    for i in range(w, -1, -1):  # 输出结果
        if states[n - 1][i]:
            return i
    return 0


def knapsack2(items, n, w):
    """
    :param items: 物品重量列表如: [2,2,4,6,3]
    :param n: 物品个数
    :param w: 背包可承载的重量 11
    :return:
    """
    states = [False for _ in range(w + 1)]  # 默认值为 False
    states[0] = True  # 第一行的值要特殊处理，可以利用哨兵优化
    if items[0] <= w:
        states[items[0]] = True

    for i in range(1, n):  # 动态规划
        for j in range(w - items[i], -1, -1):  # 把第 i 个物品放入背包，j 需要从大到小来处理。
            if states[j]:  # 如果我们按照 j 从小到大处理的话，会出现 for 循环重复计算的问题
                states[j + items[i]] = True

    for i in range(w, -1, -1):  # 输出结果
        if states[i]:
            return i
    return 0


def knapsack3(items, values, n, w):
    """
    :param items: 物品重量列表如: [2,2,4,6,3]
    :param values: 物品价值列表: [3,3,4,2,1]
    :param n: 物品个数 5
    :param w: 背包可承载的重量 9
    :return:
    """
    states = [[None for _ in range(w + 1)] for _ in range(n)]  # 默认值为 None

    # 初始化 states
    for i in range(1, n):
        for j in range(0, w + 1):
            states[i][j] = -1

    states[0][0] = 0
    if items[0] <= w:
        states[0][items[0]] = values[0]  # 初始化第一行

    for i in range(1, n):  # 动态规划，状态转移
        for j in range(0, w + 1):  # 不选择第 i 个物品
            if states[i - 1][j] >= 0:
                states[i][j] = states[i - 1][j]

        for j in range(0, w - items[i] + 1):  # 选择第 i 个物品
            if states[i - 1][j] >= 0:
                v = states[i - 1][j] + values[i]
                if v > states[i][j + items[i]]:
                    states[i][j + items[i]] = v

    # 找最大值
    max_value = -1
    for j in range(0, w + 1):
        if states[n - 1][j] > max_value:
            max_value = states[n - 1][j]
    return max_value


def double11advance(items, n, w):
    """
    :param items:商品价格列表
    :param n: 商品个数
    :param w: 表示满减条件，比如 200
    :return:
    """
    states = [[False for _ in range(3 * w + 1)] for _ in range(n)]  # 默认值为 None
    states[0][0] = True  # 第一行的数据要特殊处理
    if items[0] <= 3 * w:
        states[0][items[0]] = True

    for i in range(1, n):  # 动态规划
        for j in range(0, 3 * w + 1):  # 不购买第 i 个商品
            if states[i - 1][j]:
                states[i][j] = states[i - 1][j]
        for j in range(0, 3 * w - items[i] + 1):  # 购买第 i 个商品
            if states[i - 1][j]:
                states[i][j + items[i]] = True

    j = None
    for j in range(w, 3 * w + 1):
        if states[n - 1][j]:  # 输出结果大于等于 w 的最小值
            break

    if j >= 3 * w:  # 没有可行性解
        return

    for i in range(n - 1, 0, -1):
        if j - items[i] >= 0 and states[i - 1][j - items[i]]:  # i 表示二维数组中的行，j 表示列
            print(items[i], ":buy")  # 购买这个商品
            j = j - items[i]
        else:
            print("not buy")  # 没有购买这个商品 j_index 没有变
    if j != 0:
        print(items[0], ":buy")  # 购买这个商品


# min_dist = sys.maxint


def min_dist_bt(i, j, dist, w, n):
    """
    :param i: 行下标
    :param j: 列下标
    :param dist: 当前的距离
    :param w: 一个二维数组，记录每个格子的距离值
    :param n: 一个多少个格子
    :return: 调用方式：min_dist_bt(0, 0, 0, [[1,3,5,9],[2,1,3,4],[5,2,6,7],[6,8,4,3]], 3)
    """
    global res
    if i == n and j == n:  # 到到 n-1位置了
        current_dist = dist + w[i][j]  # 到达指定位置时的路径数据之和，这里加上了指定位置自己的值
        if current_dist < res:
            res = current_dist
            print(res)
        return
    if i < n:  # 往下走，更新 i = i+1, j = j
        min_dist_bt(i + 1, j, dist + w[i][j], w, n)
    if j < n:  # 往右走，更新 i = i, j = j + 1
        min_dist_bt(i, j + 1, dist + w[i][j], w, n)


def min_dist_dp(matrix, n):
    """
    :param matrix: 一个二维数组，记录每个格子的距离值
    :param n: 一个多少个格子
    :return: 调用方式：min_dist_dp([[1,3,5,9],[2,1,3,4],[5,2,6,7],[6,8,4,3]], 3)
    """
    states = [[0 for _ in range(n)] for _ in range(n)]
    sum_dist = 0
    for j in range(0, n):  # 初始化 states 的第一行数据
        sum_dist += matrix[0][j]
        states[0][j] = sum_dist

    sum_dist = 0
    for i in range(0, n):  # 初始化 states 的第一列数据
        sum_dist += matrix[i][0]
        states[i][0] = sum_dist

    for i in range(1, n):
        for j in range(1, n):
            states[i][j] = matrix[i][j] + min(states[i][j - 1], states[i - 1][j])
    return states[n - 1][n - 1]


# matrix = [[1, 3, 5, 9], [2, 1, 3, 4], [5, 2, 6, 7], [6, 8, 4, 3]]
# n = 4
# mem = states = [[0 for _ in range(4)] for _ in range(4)]


def min_dist_zt(i, j):
    """
    :param i: 行下标
    :param j: 列下标
    :return 调用方式：min_dist_zt(n-1, n-1)
    """
    # global matrix, n, mem

    if i == 0 and j == 0:
        return matrix[0][0]

    if mem[i][j] > 0:
        return mem[i][j]

    min_left = sys.maxint
    if j - 1 >= 0:
        min_left = min_dist_zt(i, j - 1)

    min_right = sys.maxint
    if i - 1 >= 0:
        min_right = min_dist_zt(i - 1, j)

    current_min_dist = matrix[i][j] + min(min_left, min_right)
    mem[i][j] = current_min_dist
    return current_min_dist


def lwst_bt(i, j, e_dist, a_str, b_str, min_dist):
    """
    :param i: 字符串 a_str 的下标
    :param j: 字符串 b_str 的下标
    :param e_dist: 当前的编辑距离
    :param a_str: 字符串
    :param b_str: 字符串
    :param min_dist: 记录最小的编辑距离
    :return: 回朔法计算两个字符串的编辑距离 如：lwst_bt(0,0,0,"mitcmu","mtacnu",sys.maxint)
    """
    a_str_l = len(a_str)
    b_str_l = len(b_str)
    if i == a_str_l or j == b_str_l:
        if i < a_str_l:
            e_dist += a_str_l - i
        if j < b_str_l:
            e_dist += b_str_l - j
        if e_dist < min_dist:
            min_dist = e_dist
        print(min_dist)
        return
    if a_str[i] == b_str[j]:  # 两个字符匹配
        lwst_bt(i + 1, j + 1, e_dist, a_str, b_str, min_dist)
    else:  # 两个字符不匹配
        lwst_bt(i + 1, j, e_dist + 1, a_str, b_str, min_dist)  # 删除 a_str[i] 或者在 b_str[j] 前面添加一个跟 a_str[i] 相同的字符
        lwst_bt(i, j + 1, e_dist + 1, a_str, b_str, min_dist)  # 删除 b_str[j] 或者在 a_str[i] 前面添加一个跟 b_str[j] 相同的字符
        lwst_bt(i + 1, j + 1, e_dist + 1, a_str, b_str, min_dist)  # 将 a_str[i] 和 b_str[j] 替换为相同字符


def lwst_dp(a_str, b_str):
    """
    :param a_str: 字符串
    :param b_str: 字符串
    :return:
    """
    n = len(a_str)
    m = len(b_str)
    min_dist = [[0 for _ in range(m)] for _ in range(n)]
    for j in range(0, m):  # 初始化第 0 行
        if a_str[0] == b_str[j]:
            min_dist[0][j] = j
        elif j != 0:
            min_dist[0][j] = min_dist[0][j - 1] + 1
        else:
            min_dist[0][j] = 1

    for i in range(0, n):  # 初始化第 0 列
        if b_str[0] == a_str[i]:
            min_dist[i][0] = i
        elif i != 0:
            min_dist[i][0] = min_dist[i - 1][0] + 1
        else:
            min_dist[i][0] = 1

    for i in range(1, n):  # 按行填表
        for j in range(1, m):
            if a_str[i] == b_str[j]:
                min_dist[i][j] = min(min_dist[i - 1][j] + 1, min_dist[i][j - 1] + 1, min_dist[i - 1][j - 1])
            else:
                min_dist[i][j] = min(min_dist[i - 1][j] + 1, min_dist[i][j - 1] + 1, min_dist[i - 1][j - 1] + 1)

    return min_dist[n - 1][m - 1]


def lcs(a_str, b_str):
    """ 计算编辑距离
        :param a_str: 字符串
        :param b_str: 字符串
        :return:
        """
    n = len(a_str)
    m = len(b_str)
    edit = [[i + j for j in range(len(b_str) + 1)] for i in range(len(a_str) + 1)]
    for x in range(1, n + 1):
        for y in range(1, m + 1):
            if a_str[x - 1] == b_str[y - 1]:
                d = 0
            else:
                d = 1
            edit[x][y] = min(edit[x - 1][y] + 1, edit[x][y - 1] + 1, edit[x - 1][y - 1] + d)
    return edit[len(a_str)][len(b_str)]


class Graph:
    def __init__(self, v):
        self.v = v  # 顶点个数
        self.adj = [None] * v  # 邻接表
        for i in range(0, v):
            self.adj[i] = []

    def add_edge(self, s, t):  # s 先于 t, 边 s -> t
        self.adj[s].append(t)

    def topo_sort_by_kahn(self):
        in_degree = [0] * self.v  # 统计每个顶点的入度
        for i in range(0, self.v):
            for j in range(0, len(self.adj[i])):
                w = self.adj[i][j]
                in_degree[w] += 1

        queue = deque()
        for i in range(0, self.v):
            if in_degree[i] == 0:
                queue.append(i)
        while len(queue) > 0:
            i = queue.pop()
            print("->", i)
            for j in range(0, len(self.adj[i])):
                k = self.adj[i][j]
                in_degree[k] -= 1
                if in_degree[k] == 0:
                    queue.append(k)

    def topo_sort_by_dfs(self):
        inverse_adj = [None] * self.v
        for i in range(0, self.v):
            inverse_adj[i] = []
        for i in range(0, self.v):
            for j in range(0, len(self.adj[i])):
                w = self.adj[i][j]
                inverse_adj[w].append(i)

        visited = [False] * self.v
        for i in range(0, self.v):
            if not visited[i]:
                visited[i] = True
                self.dfs(i, inverse_adj, visited)

    def dfs(self, vertex, inverse_adj, visited):
        for i in range(0, len(inverse_adj[vertex])):
            w = inverse_adj[vertex][i]
            if visited[w]:
                continue
            visited[w] = True
            self.dfs(w, inverse_adj, visited)
        print("->", vertex)


class Dijkstra:
    def __init__(self):
        self.graph = {}  # 记录地图指向关系，和相应的权重
        self.costs = {}  # 记录从起点到每个点的距离
        self.parents = {}  # 记录每个点的父节点
        self.processed = []  # 记录以及处理里的节点

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


class AStar:
    def __init__(self):
        self.graph = {}  # 记录地图指向关系，和相应的权重
        self.costs = {}  # 记录从起点到每个点的距离
        self.parents = {}  # 记录每个点的父节点
        self.processed = []  # 记录以及处理里的节点

    def add_edge(self, s, t, w, f):  # s 先于 t, 边 s -> t
        if not self.graph.get(s):
            self.graph[s] = {}
        if t and f:
            self.graph[s][t]["w"] = w
            self.graph[s][t]["f"] = f

    def dijkstra(self):
        node = self.find_lowest_f_node()  # 在未处理的节点中找出开销最小的节点
        while node is not None:  # while循环在所有节点都被处理过后结束
            cost = self.costs[node]
            neighbors = self.graph[node]
            for n in neighbors.keys():
                new_cost = cost + neighbors[n]
                if self.costs[n] > new_cost:  # 如果经当前节点前往该邻居更近
                    self.costs[n] = new_cost  # 更新该邻居节点的开销
                    self.parents[n] = node  # 该邻居节点的父节点设置为当前节点
            self.processed.append(node)  # 将当前节点标记为处理过
            node = self.find_lowest_f_node()  # 找出接下来要处理的节点，并循环
        print(self.costs)
        print(self.parents)

    def find_lowest_f_node(self):
        lowest_f = float("inf")
        lowest_f_node = None
        for node in self.costs:  # 遍历所有的节点
            f = self.costs[node]
            if f < lowest_f and node not in self.processed:
                lowest_f = f  # 就将其视为开销最低的节点
                lowest_f_node = node
        return lowest_f_node


class Bitmap:
    def __init__(self, max_value):
        """确定所需数组个数"""
        self.size = int((max_value + 31 - 1) / 31)  # 初始化bitmap 向上取整
        self.array = [0 for _ in range(self.size)]

    def set_1(self, num):
        """将元素所在的位置1"""
        elem_index = num / 31  # 计算在数组中的索引
        byte_index = num % 31  # 计算在数组中的位索引
        ele = self.array[elem_index]
        self.array[elem_index] = ele | (1 << byte_index)  # 相关位置1, 左移 byte_index 位相当于乘以2的 byte_index 次方

    def test_1(self, num):
        """检测元素存在的位置"""
        elem_index = num / 31
        byte_index = num % 31
        if self.array[elem_index] & (1 << byte_index):
            return True
        return False


def fn(a):
    return a % 2 == 1


class Singleton(object):
    __instance = None

    def __new__(cls, age, name):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance


class TimeException(Exception):
    def __init__(self, exception_info):
        super.__init__()
        self.info = exception_info

    def __str__(self):
        return self.info


def time_check(func):
    import datetime

    def wrapper(*args, **kwargs):
        if datetime.datetime.now().year == 2019:
            func(*args, **kwargs)
        else:
            raise TimeException("函数已过时")

    return wrapper


def find_2(a_list, target):
    low = 0
    high = len(a_list) - 1
    mid = (low + high) / 2
    while low <= high:
        if a_list[mid] == target:
            return mid
        if target < a_list[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n


def _not_divisible(n):
    return lambda x: x % n > 0


def primes():
    yield 2
    it = _odd_iter()  # 初始序列
    while True:
        n = next(it)  # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it)  # 构造新序列


def is_palindrome(n):
    _length = len(str(n))
    if _length % 2 == 1:
        return str(n)[0:_length / 2] == str(n)[_length / 2 + 1:][::-1]
    else:
        return str(n)[0:_length / 2] == str(n)[_length / 2:][::-1]


# def createCounter():
#     def seed():
#         s = 0
#         while True:
#             s += 1
#             yield s
#
#     _iter = seed()
#     def counter():
#         return next(_iter)
#     return counter

# def createCounter():
#
#     def counter1():
#         s = 0
#         while True:
#             s += 1
#             yield s
#
#     test = counter1()
#
#     def counter2():
#         return next(test)
#
#     return counter2
# import asyncio
# # from aiohttp import web
#
# async def index(request):
#     await asyncio.sleep(0.5)
#     return web.Response(body=b'<h1>Index</h1>')
#
# async def hello(request):
#     await asyncio.sleep(0.5)
#     text = '<h1>hello, %s!</h1>' % request.match_info['name']
#     return web.Response(body=text.encode('utf-8'))
#
# async def init(loop):
#     app = web.Application(loop=loop)
#     app.router.add_route('GET', '/', index)
#     app.router.add_route('GET', '/hello/{name}', hello)
#     srv = await loop.create_server(app.make_handler(), '127.0.0.1', 8000)
#     print('Server started at http://127.0.0.1:8000...')
#     return srv

# loop = asyncio.get_event_loop()
# loop.run_until_complete(init(loop))
# loop.run_forever()
import os
# 将字符串的时间转换为时间戳
import time


class SingleTong(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(SingleTong, cls).__new__(cls, *args, **kwargs)
        return cls._instance


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def log(text):
    def decorate(func):
        def wrapper(*args, **kwargs):
            print("%s fuck you" % text)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def bin_or_oct_to_dec(num, base):
    """二进制, 八进制, 转十进制, base =2,8"""
    result = 0
    length = len(num)
    for x in range(length):
        result += base ** x * int(num[length - x - 1])
    return result
    # return int(num,2)
    # return int(num,8)


def hex_to_dec(num):
    """16进制转 10进制 """
    base = [str(x) for x in range(10)] + [chr(x) for x in
                                          range(ord('A'), ord("A") + 6)]  # 前者把 0 ~ 9 转换成字符串存进列表 base 里，后者把 A ~ F 存进列表
    # 相当于 base = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    result = 0
    length = len(num)
    for x in range(length):
        result += 16 ** x * int(base.index(num[length - x - 1]))
    return result
    # return int(num,16)


def dec_to_bin_or_oct(num, base):
    """十进制转二进制 或八进制 base=2 或者 8"""
    l = []  # 创建一个空列表
    if num < 0:  # 是负数转换成整数
        return "-" + dec_to_bin_or_oct(abs(num), base)  # 如过是负数，先转换成正数
    while True:
        num, reminder = divmod(num, base)  # 短除法，对2求，分别得到除数 和 余数、这是 Python 的特有的一个内置方法，分别可以到商 及 余数
        l.append(str(reminder))  # 把获得的余数 存入字符串
        if num == 0:  # 对应了前面的话，当商为 0时，就结束啦
            return "".join(l[::-1])  # 对列表中的字符串进行逆序拼接，得到一个二进制字符串
            # return bin(num)
            # return oct(num)


def dec_to_hex(num):
    """十进制转十六进制（这个相对麻烦一点，因为，十六进制包含 A-F，大小写不敏感）"""
    base = [str(x) for x in range(10)] + [chr(x) for x in
                                          range(ord('A'), ord("A") + 6)]  # 前者把 0 ~ 9 转换成字符串存进列表 base 里，后者把 A ~ F 存进列表
    # 相当于 base = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    l = []
    if num < 0:
        return "-" + dec_to_hex(abs(num))
    while True:
        num, rem = divmod(num, 16)  # 求商 和 留余数
        l.append(base[rem])
        if num == 0:
            return "".join(l[::-1])
            # return hex(num)


import sys

import sys


def ip2bin(mask):
    """
    ip地址转换为二进制数
    :param mask:
    :return: string类型
    """
    l = list()
    for i in mask:
        _ip = bin(i)[2:]
        if len(str(_ip)) < 8:
            _ip = "0" * (8 - len(str(_ip))) + str(_ip)
        l.extend(_ip)
    return ''.join(str(j) for j in l)


from collections import OrderedDict

test = OrderedDict()


def check_mask(mask):
    """

    :param mask: 存储mask四个十进制位的列表
    :return: mask合法返回True
    """
    if mask in ([255, 255, 255, 255], [0, 0, 0, 0]):
        return False
    b_ip = ip2bin(mask)
    flag = False  # 是否碰到了0
    for i, c in enumerate(b_ip):
        if flag:
            if c == '1':
                return False
        else:
            if c == '0':
                flag = True
    else:
        return True


def check_ip(ip):
    """

    :param ip: 存储ip四个十进制位的列表
    :return: ip合法返回True
    """
    for i in ip:
        try:
            temp = int(i)
        except:
            return False
        if temp < 0 or temp > 255:
            return False
    else:
        return True


def longestPalindrome(s):
    if s == s[::-1]:
        return len(s)

    maxLen = 0

    for i in range(len(s)):
        a_index = i - maxLen
        if a_index >= 1 and s[a_index - 1:i + 1] == s[a_index - 1:i + 1][::-1]:
            maxLen += 2
            continue
        if a_index >= 0 and s[a_index:i + 1] == s[a_index:i + 1][::-1]:
            maxLen += 1

    return maxLen


def get_all_rabits(n):
    if n in [1, 2]:
        return 1
    if n == 3:
        return 2
    else:
        res = get_all_rabits(n - 1) + 2 * get_all_rabits(n - 4)
        return res


def check_ip_is_ok(ip):
    try:
        ip_arr = str(ip).split(".")
        for i in ip_arr:
            if int(i) < 0 or int(i) > 255:
                return False
        return True
    except:
        return False


def check_mask_is_ok(mask):
    try:
        mask_arr = str(mask).split(".")
        for i in mask_arr:
            bin_str = bin(i).replace("0b", "")
            if len(bin_str) < 8 and "1" in str(bin_str):
                return False
        return True
    except:
        return False


class Node(object):
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    def __init__(self, node=None):
        self._head = node

    def length(self):
        cur = self._head
        count = 0
        while cur:
            cur = cur.next
            count += 1
        return count

    def travel(self):
        cur = self._head
        while cur:
            print(cur.elem, end=" ")
            cur = cur.next
        print("")

    def add(self, item):
        node = Node(item)
        node.next = self._head
        self._head = node

    def insert(self, item, index):
        node = Node(item)
        if not self._head:
            self._head = node
        else:
            cur = self._head
            while cur:
                if cur.elem == index:
                    node.next = cur.next
                    cur.next = node
                    break
                else:
                    cur = cur.next

    def remove(self, item):
        cur = self._head
        pre = None
        while cur:
            if cur.elem == item:
                if cur == self._head:
                    self._head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

            class Node(object):
                def __init__(self, data):
                    self.data = data
                    self.next = None

            class SingleLinkList(object):
                def __init__(self, node=None):
                    self._head = node

                def add(self, data):
                    node = Node(data)
                    node.next = self._head
                    self._head = node

                def length(self):
                    cur = self._head
                    count = 0
                    while cur:
                        count += 1
                        cur = cur.next
                    return count

                def traval(self):
                    cur = self._head
                    while cur:
                        print(cur.data, end=" ")
                        cur = cur.next
                    print("")

                def insert(self, data, target_data):
                    node = Node(data)
                    if not self._head:
                        self._head = node
                    else:
                        cur = self._head
                        while cur:
                            if cur.data == target_data:
                                node.next = cur.next
                                cur.next = node
                                break
                            else:
                                cur = cur.next

                def remove(self, target_data):
                    cur = self._head
                    pre = None
                    while cur.next:
                        if cur.data == target_data:
                            if cur == self._head:
                                self.head = cur.next
                            else:
                                pre.next = cur.next
                            break
                        else:
                            pre = cur
                            cur = cur.next

                a = sys.stdin.readlines().split(" ")
                a = list(map(int, a))
                node_num = a[0]
                head_node = Node(a[1])
                nodes = a[2:len(a) - 1]
                to_del_data = a[-1]
                single_link_list = SingleLinkList(head_node)
                for i in range(0, len(nodes), 2):
                    data = nodes[i]
                    target_data = nodes[i + 1]
                    single_link_list.insert(data, target_data)

                single_link_list.remove(to_del_data)
                single_link_list.traval()


from threading import Thread, Condition

res = []
condition = Condition()
current = "A"


def send_a(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "A":
                condition.wait()
            res.append("A")
            current = "B"
            condition.notify_all()


def send_b(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "B":
                condition.wait()
            res.append("B")
            current = "C"
            condition.notify_all()


def send_c(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "C":
                condition.wait()
            res.append("C")
            current = "D"
            condition.notify_all()


def send_d(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "D":
                condition.wait()
            res.append("D")
            current = "A"
            condition.notify_all()


# if __name__ == '__main__':
#     a = 1453
#     thread_list = [Thread(target=send_a, args=(a,)), Thread(target=send_b, args=(a,)), Thread(target=send_c, args=(a,)),
#                    Thread(target=send_d, args=(a,))]
#     for thread in thread_list:
#         thread.start()
#
#     for thread in thread_list:
#         thread.join()
#     print("".join(res))

import re


def calculate(n1, n2, operator):
    if operator == "+":
        result = n1 + n2
    elif operator == "-":
        result = n1 - n2
    elif operator == "*":
        result = n1 * n2
    elif operator == "/":
        result = n1 / n2
    else:
        raise Exception("operator is not support now...")
    return result


def is_operator(e):
    operators = ["+", "-", "*", "/", "(", ")"]
    return True if e in operators else False


def formula_format(formula):
    """
    步骤需要处理的是区分横杠‘-’是代表负数还是减号
    """

    formula = re.sub(' ', '', formula)

    # 以 '横杠数字' 分割， -> \- 表示匹配横杠开头； \d+ 表示匹配数字1次或多次；\.?表示匹配小数点0次或1次;\d*表示匹配数字1次或多次。
    formula_list = [i for i in re.split('(\-\d\.?\d*)', formula) if i]

    final_formula = []
    for item in formula_list:

        # 第一个是以横杠开头的数字（包括小数）final_formula。即第一个是负数，横杠就不是减号
        if len(final_formula) == 0 and re.search('^\-\d+\.?\d*$', item):
            final_formula.append(item)
            continue

        if len(final_formula) > 0:

            # 如果final_formal最后一个元素是运算符['+', '-', '*', '/', '('], 则横杠数字不是负数
            if re.search('[\+\-\*\/\(]$', final_formula[-1]):
                final_formula.append(item)
                continue

        # 剩下的按照运算符分割开
        item_split = [i for i in re.split('([\+\-\*\/\(\)])', item) if i]
        final_formula += item_split

    return final_formula


def decision(tail_op, now_op):
    """
    :param tail_op: 运算符栈最后一个运算符
    :param now_op: 算式列表取出当前运算符
    :return: 1 弹栈， 0 弹出运算符栈最后一个元素， -1 入栈
    """
    # 运算符等级
    rate1 = ["+", "-"]
    rate2 = ["*", "/"]
    rate3 = ["("]
    rate4 = [")"]

    if tail_op in rate1:
        if now_op in rate2 or now_op in rate3:
            # 运算符优先级不同
            return -1  # 把当前取出的运算符压栈 "1+2+3"
        else:
            return 1  # 否则运算符栈中最后的 运算符弹出，进行计算

    elif tail_op in rate2:
        if now_op in rate3:
            return -1
        else:
            return 1

    elif tail_op in rate3:
        if now_op in rate4:
            return 0  # ( 遇上 ) 需要弹出 (，丢掉 )
        else:
            return -1  # 只要栈顶元素为(，当前元素不是)都应入栈
    else:
        return -1


# 主函数  ->  遍历算式列表中的字符，决定入栈弹栈操作
def final_cal(formula_list):
    num_stack = []
    op_stack = []
    for e in formula_list:
        operator = is_operator(e)
        if not operator:
            # 数字栈 ->  转为浮点数
            a = 2
            num_stack.append(float(int(e)))
        else:
            # e 开始进入 运算符栈, -> 判断
            while True:
                a = 1
                if len(op_stack) == 0:  # 第一个运算符来了，都得入栈
                    op_stack.append(e)
                    break
                # 后面运算符来了，需要判断入栈，or 出栈。
                pop_oper = op_stack[-1]
                tag = decision(op_stack[-1], e)
                if tag == -1:  # 压栈
                    op_stack.append(e)
                    break
                elif tag == 0:  # 弹出运算符栈内最后一个 "("， 丢掉当前的 ")", 进入下次循环
                    op_stack.pop()
                    break
                elif tag == 1:  # 运算符栈弹出最后一个运算符，数字栈弹出最后两个元素，进行计算
                    op = op_stack.pop()
                    num2 = num_stack.pop()
                    num1 = num_stack.pop()

                    # 计算后结果 --> 压入数字栈
                    num_stack.append(calculate(num1, num2, op))

    # 处理大循环结束后 数字栈和运算符栈中可能还有元素 的情况
    while len(op_stack) != 0:
        op = op_stack.pop()
        num2 = num_stack.pop()
        num1 = num_stack.pop()
        num_stack.append(calculate(num1, num2, op))
    return num_stack, op_stack


class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None


class SingleLinkedList(object):
    def __init__(self):
        self._head = None
        self.count = 0

    def add(self, node):
        if not self._head:
            self._head = node
        else:
            node.next = self._head
            self._head = node
        self.count += 1

    def get_node_by_index(self, index):
        if index > self.count:
            print(None)
        else:
            i = 0
            cur = self._head
            while i < self.count - index:
                cur = cur.next
                i += 1
            print(cur.data)


def get_first_place_even(n):
    arr = [[0] * (2 * n - 1) for n in range(1, n + 1)]
    arr[1][1] = 1
    arr[0][0] = 1
    arr[1][2] = 1
    arr[1][0] = 1
    for i in range(2, n):
        arr[i][0] = 1
        for j in range(1, 2 * i):
            if j - 2 < 0:
                test1 = 0
            else:
                test1 = arr[i - 1][j - 2]

            if len(arr[i - 1]) <= j:
                test2 = 0  # -*- coding: utf-8 -*-


# import re
import functools
import sys
from collections import deque


class Sort(object):
    """常见的排序算法"""

    def __init__(self):
        pass

    @staticmethod
    def bubble_sort(a_list):
        """冒泡排序"""
        list_length = len(a_list)
        if list_length <= 1:
            return a_list

        list_length = len(a_list)
        for x in range(0, list_length):
            flag = False
            for y in range(0, list_length - x - 1):
                if a_list[y] > a_list[y + 1]:
                    temp = a_list[y]
                    a_list[y] = a_list[y + 1]
                    a_list[y + 1] = temp
                    flag = True
            if not flag:
                return a_list

    @staticmethod
    def insert_sort(a_list):
        """插入排序"""
        list_length = len(a_list)
        if list_length <= 1:
            return a_list

        for x in range(1, list_length):
            to_insert_value = a_list[x]
            y = x - 1

            while y >= 0:
                if to_insert_value < a_list[y]:
                    a_list[y + 1] = a_list[y]
                else:
                    break
                y -= 1
            a_list[y + 1] = to_insert_value
        return a_list

    @staticmethod
    def select_sort(a_list):
        """选择排序"""
        list_length = len(a_list)
        if list_length <= 1:
            return a_list
        for x in range(0, list_length):
            min_index = x
            for y in range(x + 1, list_length):
                if a_list[y] < a_list[min_index]:
                    min_index = y

            if min_index != x:
                a_list[x], a_list[min_index] = a_list[min_index], a_list[x]
        return a_list

    def merge_sort(self, a_list):
        """归并排序"""
        if len(a_list) <= 1:
            return a_list

        mid = len(a_list) / 2
        left = self.merge_sort(a_list[:mid])
        right = self.merge_sort(a_list[mid:])

        return self._merge(left, right)

    @staticmethod
    def _merge(left, right):
        """归并排序之合并"""
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:]
        result += right[j:]
        return result

    def easy_quick_sort(self, a_list):
        """快速排序简单实现"""
        if len(a_list) < 2:
            return a_list

        pivot = a_list[0]

        less_than_pivot = [x for x in a_list if x < pivot]
        more_than_pivot = [x for x in a_list if x > pivot]
        return self.easy_quick_sort(less_than_pivot) + [pivot] + self.easy_quick_sort(more_than_pivot)

    def quick_sort(self, a_list):
        """快速排序"""
        return self._quick_sort(a_list, 0, len(a_list) - 1)

    def _quick_sort(self, a_list, left, right):
        if left < right:
            pivot = self._partition(a_list, left, right)
            self._quick_sort(a_list, left, pivot - 1)
            self._quick_sort(a_list, pivot + 1, right)
        return a_list

    @staticmethod
    def _partition(a_list, left, right):
        pivot_key = a_list[left]
        while left < right:
            while left < right and a_list[right] >= pivot_key:
                right -= 1
            a_list[left] = a_list[right]

            while left < right and a_list[left] <= pivot_key:
                left += 1
            a_list[right] = a_list[left]
        a_list[left] = pivot_key
        return left

    @staticmethod
    def bucket_sort(a_list):
        """桶排序，支持负数和2位小数"""
        if not a_list or len(a_list) < 2:
            return a_list

        accuracy = 1
        _max = int(max(a_list) * accuracy)
        _min = int(min(a_list) * accuracy)
        buckets = [0 for _ in range(_min, _max + 1)]

        for x in a_list:
            buckets[int(x) * accuracy - _min] += 1

        current = _min
        n = 0
        for bucket in buckets:
            while bucket > 0:
                a_list[n] = current / accuracy
                bucket -= 1
                n += 1
            current += 1
        return a_list

    @staticmethod
    def count_sort(a_list):
        """计数排序, 不支持负数"""
        if not a_list or len(a_list) < 2:
            return a_list

        _max = max(a_list)
        buckets = [0 for _ in range(0, _max + 1)]
        for x in a_list:
            buckets[x] += 1

        for x in range(1, _max + 1):
            buckets[x] += buckets[x - 1]
        result = [0 for _ in range(0, len(a_list))]

        for x in range(0, len(a_list)):
            index = buckets[a_list[x]] - 1
            result[index] = a_list[x]
            buckets[a_list[x]] -= 1
        return result

    @staticmethod
    def radix_sort(a_list):
        """基数排序"""
        i = 0
        _max = max(a_list)
        j = len(str(_max))  # 记录最大值的位数
        while i < j:
            bucket_list = [[] for _ in range(10)]
            for x in a_list:
                bucket_list[int(x / (10 ** i)) % 10].append(x)
            a_list = [y for x in bucket_list for y in x]
            i += 1
        return a_list


class Find(object):
    """常见的查找算法"""

    def __init__(self):
        pass

    @staticmethod
    def b_search(a_list, target):
        """二分查找算法 非递归实现"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target == a_list[mid]:
                return mid
            if target < a_list[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def b_recursion_search(self, a_list, target, low, high):
        """二分查找算法 非递归实现"""
        if low > high:
            return -1
        mid = low + (high - low) / 2
        if target == a_list[mid]:
            return mid
        if target < a_list[mid]:
            return self.b_recursion_search(a_list, target, low, mid - 1)
        else:
            return self.b_recursion_search(a_list, target, mid + 1, high)

    @staticmethod
    def b_first_search(a_list, target):
        """查找第一个等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target <= a_list[mid]:
                if target == a_list[mid] and target != a_list[mid - 1]:
                    return mid
                else:
                    high = mid - 1
            else:
                low = mid + 1
        return -1

    @staticmethod
    def b_last_search(a_list, target):
        """查找最后一个等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target >= a_list[mid]:
                if target == a_list[mid] and target != a_list[mid + 1]:
                    return mid
                else:
                    low = mid + 1
            else:
                high = mid - 1
        return -1

    @staticmethod
    def b_last_than_search(a_list, target):
        """查找最后一个大于等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target <= a_list[mid]:
                if mid == 0 or target > a_list[mid - 1]:
                    return mid
                else:
                    high = mid - 1
            else:
                low = mid + 1
        return -1

    @staticmethod
    def b_last_less_search(a_list, target):
        """查找最后一个小于等于目标值"""
        low = 0
        high = len(a_list) - 1
        while low <= high:
            mid = low + (high - low) / 2
            if target >= a_list[mid]:
                if mid == len(a_list) - 1 or target < a_list[mid + 1]:
                    return mid
                else:
                    low = mid + 1
            else:
                high = mid - 1
        return -1

    @staticmethod
    def sqrt(x):
        """求一个数的算数平方根，精确到小数点后6位"""
        low = 0
        mid = x / 2.0
        high = x
        while abs(mid ** 2 - x) > 0.000001:
            if mid ** 2 < x:
                low = mid
            else:
                high = mid
            mid = (low + high) / 2
        return round(mid, 6)


def generate_bc(sub_str):
    """
    :param sub_str: 是模式串
    :return: 返回一个散列表  散列表的下标是 b_char 中每个字符的 ascii 码值，下标对应的值是该字符在b_char中最后一次出现的位置
    """
    ascii_size = 256
    ascii_list = [-1] * ascii_size  # 初始化一个散列表

    for i in range(0, len(sub_str)):
        ascii = ord(sub_str[i])  # 计算 b_char中每个字符的 ascii 值
        ascii_list[ascii] = i  # 存每个字符在 b_char 最后一次出现的位置
    return ascii_list


def bm_simple(main_str, sub_str):
    """
    :param main_str: 主字符串
    :param sub_str: 模式串
    :return:
    """
    bc = generate_bc(sub_str)  # 记录模式串中每个字符最后出现的位置，也就是坏字符的哈希表
    i = 0  # 表示主串和模式串对齐的第一个字符
    n = len(main_str)  # 表示主串的长度
    m = len(sub_str)  # 表示子串的长度
    while i <= n - m:
        j = m - 1
        while j >= 0:
            if main_str[i + j] != sub_str[j]:  # 坏字符对应模式串中的下标是 j
                break
            j -= 1
        if j < 0:
            return i  # 匹配成功，返回主串和模式串第一个匹配的字符的位置
        bad_str_index = bc[ord(main_str[i + j])]  # 坏字符在模式串中的位置
        i += (j - bad_str_index)  # 这里等同于将模式串往后滑动  j - bc[ord(main_str[i + j]) 位
    return -1


def generate_gs(sub_str):
    m = len(sub_str)
    suffix = [-1] * m
    prefix = [False] * m
    for i in range(0, m - 1):
        j = i
        k = 0  # 公共后缀串长度，
        while j >= 0 and sub_str[j] == sub_str[m - 1 - k]:  # 求公共后缀子串，先计算第一个跟模式子串最后一个字符相匹配的位置
            j -= 1  # 然后依次比较前面的字符是否相等，比如：cabcab，先从前往后历遍，计算得到 b 字符是符合要求的，
            k += 1  # 此时再从 b 字符前一个位置与模式串倒数第二个位置的字符去比较，如果还是相等，则继续，循环了 k 次说明匹配的字符长度就是 k
            suffix[k] = j + 1  # j+1 表示公共后缀子串在 b[0, i]中的起始下标

        if j == -1:  # 如果公共后缀子串也是模式串的前缀子串
            prefix[k] = True
    return suffix, prefix


def move_by_gs(j, m, suffix, prefix):
    """j 表示坏字符对应的模式串中的字符下标，m 表示模式串的长度"""
    k = m - 1 - j  # k 好后缀长度
    if suffix[k] != -1:
        return j - suffix[k] + 1
    for r in range(j + 2, m):
        if prefix[m - r]:
            return r
    return m  # 如果前面两个规则都不使用，则直接往后滑动 m 位


def bm(main_str, sub_str):
    """
    :param main_str: 主字符串
    :param sub_str: 模式串
    :return:
    """
    bc = generate_bc(sub_str)  # 记录模式串中每个字符最后出现的位置，也就是坏字符的哈希表
    suffix, prefix = generate_gs(sub_str)

    i = 0  # 表示主串和模式串对齐的第一个字符
    n = len(main_str)  # 表示主串的长度
    m = len(sub_str)  # 表示子串的长度
    while i <= n - m:
        j = m - 1
        while j >= 0:
            if main_str[i + j] != sub_str[j]:  # 坏字符对应模式串中的下标是 j
                break
            j -= 1
        if j < 0:
            return i  # 匹配成功，返回主串和模式串第一个匹配的字符的位置

        x = j - bc[ord(main_str[i + j])]  # 计算坏字符规则每次需要往后滑动的次数
        y = 0
        if j < m - 1:  # 如果有好后缀的话
            y = move_by_gs(j, m, suffix, prefix)  # 计算好后缀规则每次需要往后滑动的次数
        i = i + max(x, y)

    return -1


def get_next_list(sub_str):
    """计算如： abababzabababa 每个位置的前缀集合与后缀集合的交集中最长元素的长度
       输出为： [-1, -1, 0, 1, 2, 3, -1, 0, 1, 2, 3, 4, 5, 4]
       计算最后一个字符 a 的逻辑如下：
       abababzababab 前缀后缀最大匹配了 6 个（ababab），次大匹配是4 个（abab），次大匹配的前缀后缀只可能在 ababab 中，
       所以次大匹配数就是 ababab 的最大匹配数，在数组中查到出该值为 3。第三大的匹配数同理，它既然比 3 要小，那前缀后缀也只能在 abab 中找，
       即 abab 的最大匹配数，查表可得该值为 1。再往下就没有更短的匹配了。来计算最后位置 a 的值：既然末尾字母不是 z，
       那么就不能直接 5+1=7 了，我们回退到次大匹配 abab，刚好 abab 之后的 a 与末尾的 a 匹配，所以 a 处的最大匹配数为 4。
    """
    m = len(sub_str)
    next_list = [None] * m
    next_list[0] = -1
    k = -1
    i = 1
    while i < m:
        while k != -1 and sub_str[k + 1] != sub_str[i]:  # k+1 是每个前缀子串的最后一个字符，i 是每个后缀子串的第一个字符
            k = next_list[k]  # 求次大匹配数，次大匹配数如 abab 是上层 ababab 的最大匹配数
        if sub_str[k + 1] == sub_str[i]:  # 如果前缀子串的最后一个字符和后缀子串的第一个字符相等，则增加前缀子串和后缀子串的长度再比较
            k += 1
        next_list[i] = k  # sub_str[0:i] 的前缀集合与后缀集合的交集中最长元素的长度
        i += 1
    return next_list


def kmp(main_str, sub_str):
    n = len(main_str)
    m = len(sub_str)
    next_list = get_next_list(sub_str)
    i = 0
    j = 0
    while i < n and j < m:
        if j == -1 or main_str[i] == sub_str[j]:  # 如果子串的每个字符和主串的每个字符相等就继续循环
            i += 1
            j += 1
        else:  # 在模式串j处失配，需要找到sub_str[0:j-1]的前缀集合与后缀集合的交集中最长元素的长度, 然后直接将j更新
            j = next_list[j]  # 为next_list[j]，本来是需要将main_str[i] 处的字符依次与 sub_str[0:j]中的每个元素依次再比较的
    if j == m:
        return i - j
    return -1


class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.Trie = {}

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curr = self.Trie
        for w in word:
            if w not in curr:
                curr[w] = {}
            curr = curr[w]
        curr['#'] = 1
        print(self.Trie)

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self.Trie
        for w in word:
            if w not in curr:
                return False
            curr = curr[w]
        return "#" in curr

    def starts_with(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curr = self.Trie
        for w in prefix:
            if w not in curr:
                return False
            curr = curr[w]
        return True


class TrieNode(object):
    __slots__ = ['value', 'next_node', 'fail', 'emit']  # 限制实例的属性只能是这四个

    def __init__(self, value):
        self.value = value
        self.next_node = dict()
        self.fail = None
        self.emit = None  # 当到达叶子节点时记录此时的路径，也就是一个完整的字符串


class AcAutoTrie(object):
    __slots__ = ['_root']  # 限制实例的属性只有一个

    def __init__(self, words):
        self._root = self._build_trie(words)
        self._build_fail()

    @staticmethod
    def _build_trie(words):
        """使用 words 构建 trie 树"""
        assert isinstance(words, list) and words
        root = TrieNode('root')
        for word in words:
            node = root
            for c in word:
                if c not in node.next_node:  # 如果字符在子节点中不存在，则以字符 c 为value 生成一个 TrieNode
                    node.next_node[c] = TrieNode(c)  # 并以字符 c 为 key 放入当前 node 的子节点中
                node = node.next_node.get(c)  # 并将当前的 node 更新为 以 c 为value 生成一个 TrieNode
            node.emit = word  # 历遍完一个 word 之后，说明到底树的叶子节点了

        return root

    def _build_fail(self):
        """构建失败指针"""
        queue = []
        queue.insert(0, (self._root, None))  # 将 root 节点和 root 节点的父节点放入到队列中
        while len(queue) > 0:
            node_parent = queue.pop()
            curr, parent = node_parent[0], node_parent[1]  # 取出队列中的一个 node 和其父 node
            for sub in curr.next_node.itervalues():  # 历遍当前 node  所有子 node 全部放入队列中
                queue.insert(0, (sub, curr))
            if parent is None:  # 如果当前节点是 root 节点则跳过循环
                continue
            elif parent is self._root:  # 如果当前节点的父节点是 root 则将当前节点的失败指针指向 root
                curr.fail = self._root
            else:
                fail = parent.fail  # 否则就一直找啊找, 这里的 fail 其实很多时候等于 root 的
                while fail and curr.value not in fail.next_node:
                    fail = fail.fail
                if fail:  # 直到有一个失败指针指向的节点假设叫X吧，当前节点的值在X的子节点里存在
                    curr.fail = fail.next_node.get(curr.value)  # 那么就把当前节点的失败指针指向X的那个与当前节点相等的子节点
                else:
                    curr.fail = self._root  # 如果全部都都找完了也没找到，那就把当前节点的失败指针指向 root

    def search(self, s):
        """AC 自动的搜索匹配"""
        seq_list = []  # 用来保存返回结果的列表
        node = self._root  # 从 root 节点开始
        for i, c in enumerate(s):  # 历遍字符串 s 里的每个字符，其中 i 是字符 c 在 s 里的索引
            while c not in node.next_node and node != self._root:  # 一直循环去查找，直到字符 c 在当前节点的子节点中存在，
                node = node.fail  # 或者当前节点是 root 节点了就停止查找
            node = node.next_node.get(c)  # 将当前的 node 更新为等于它自己的值等于 c 的那个子 node
            if not node:
                node = self._root  # 如果不存在这么一个子 node 那上面的 while 循环肯定是满足 node = self._root 的
            temp = node
            while temp != self._root:  # 如果此时字符 c 不匹配任何当前 node 的子 node 则跳过下面的循环，继续去找下一个字符
                if temp.emit:  # 否则，如果循环到字符 c 时候已经到了树的叶子节点了，说明找到了匹配的字符
                    from_index = i - len(temp.emit) + 1  # 计算匹配到的字符的起始位置
                    match_info = (from_index, temp.emit)
                    seq_list.append(match_info)
                temp = temp.fail
        return seq_list


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


class Backtracking(object):
    """回朔算法案列分析"""

    def __init__(self):
        self.result = [None] * 8  # 存储 8 皇后问题的结果，下标表示行，值表示皇后存储的列
        self.count = 0  # 八皇后所有可能排列位置数

    def cal_8_queens(self, row):
        if row == 8:  # 8个棋子都放置好了，打印结果
            self.count += 1
            self.print_queens()
            print('-' * 20)
            return

        for column in range(0, 8):  # 每一行有 8个放法
            if self.is_column_ok(row, column):
                self.result[row] = column  # 第 row 行的棋子放在 column 列
                self.cal_8_queens(row + 1)  # 计算下一个

    def is_column_ok(self, row, column):
        left_column = column - 1
        right_column = column + 1
        for i in range(row - 1, -1, -1):  # 逐行往上考察
            if self.result[i] in [column, left_column, right_column]:  # 检查上一行,对角线上, 右上对角线上是否有皇后
                return False
            left_column -= 1
            right_column -= 1
            i -= 1
        return True

    def print_queens(self):
        for row in range(0, 8):
            column_str = ""
            for column in range(0, 8):
                if self.result[row] == column:
                    column_str += "Q "
                else:
                    column_str += "* "
            print(column_str)


class Patten:
    """回朔法实现正则表达式"""

    def __init__(self, patten):
        self.matched = True
        self.patten = patten  # 正则表达式
        self.p_len = len(patten)  # 正则表达式长度

    def match(self, text):
        self.matched = False
        self.re_match(0, 0, text)
        return self.matched

    def re_match(self, text_index, patten_index, text):
        if self.matched:
            return
        if patten_index == self.p_len:  # 正在表达式到结尾了
            if text_index == len(text):  # 文本串也到结尾了
                self.matched = True
            return
        if self.patten[patten_index] == "*":  # 匹配任意字符
            for k in range(0, len(text) - text_index + 1):
                self.re_match(text_index + k, patten_index + 1, text)
        elif self.patten[patten_index] == "?":  # 0 个或一个字符
            self.re_match(text_index, patten_index + 1, text)
            self.re_match(text_index + 1, patten_index + 1, text)

        elif text_index < len(text) and self.patten[patten_index] == text[text_index]:  # 纯字符匹配才行
            self.re_match(text_index + 1, patten_index + 1, text)


class Backpack:
    def __init__(self):
        self.max_weight = -1  # tracking the max weight
        self.max_value = -1  # tracking the max value

    def backpack(self, i, current_weight, items, limited_weight):
        """
        只满足背包重量最大
        0-1 背包 问题
        :param i: the ith item, integer
        :param cw:  current weight, integer
        :param items:  python list of item weights
        :param w: upper limit weight the backpack can load
        :return:
        """

        if current_weight == limited_weight or i == len(items):  # base case
            if current_weight > self.max_weight:
                self.max_weight = current_weight
            return

        self.backpack(i + 1, current_weight, items, limited_weight)  # 递归调用表示不选择当前物品，直接考虑下一个（第 i+1 个），故 cw 不更新
        if current_weight + items[i] <= limited_weight:
            self.backpack(i + 1, current_weight + items[i], items,
                          limited_weight)  # 递归调用表示选择了当前物品，故考虑下一个时，cw 通过入参更新为 cw + items[i]

    def backpack2(self, i, current_weight, items, limited_weight):
        """
        只满足背包重量最大,用一个二维数组记录之前已经计算的值，避免重复计算
        0-1 背包 问题
        :param i: the ith item, integer
        :param cw:  current weight, integer
        :param items:  python list of item weights
        :param w: upper limit weight the backpack can load
        :return:
        """
        mem = [[False for _ in range(limited_weight + 2)] for _ in range(len(items) + 1)]
        if current_weight == limited_weight or i == len(items):  # base case
            if current_weight > self.max_weight:
                self.max_weight = current_weight
            return
        if mem[i][current_weight]:
            return
        mem[i][current_weight] = True
        self.backpack2(i + 1, current_weight, items, limited_weight)  # 递归调用表示不选择当前物品，直接考虑下一个（第 i+1 个），故 cw 不更新
        if current_weight + items[i] <= limited_weight:
            self.backpack2(i + 1, current_weight + items[i], items,
                           limited_weight)  # 递归调用表示选择了当前物品，故考虑下一个时，cw 通过入参更新为 cw + items[i]

    def backpack3(self, i, limited_weight, current_weight, current_value, items, values):
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
        if current_weight == limited_weight or i == len(items):  # cw == w 表示装满了， i==n 表示物品都考察完了
            if current_value > self.max_value:
                self.max_value = current_value
            return

        self.backpack3(i + 1, limited_weight, current_weight, current_value, items,
                       values)  # 递归调用表示不选择当前物品，直接考虑下一个（第 i+1 个），故 cw 不更新
        if current_weight + items[i] <= limited_weight:
            # 递归调用表示选择了当前物品，故考虑下一个时，cw 通过入参更新为 cw + items[i]
            self.backpack3(i + 1, limited_weight, current_weight + items[i], current_value + values[i], items, values)

    @staticmethod
    def backpack4():
        """有依赖情况的背包问题"""
        N, M = 3200, 60
        f = [0] * N
        # 分组背包，每组有四种情况，a.主件 b.主件+附件1 c.主件+附件2 d.主件+附件1+附件2
        v = [[0 for i in range(4)] for j in range(M)]  # 金额
        w = [[0 for i in range(4)] for j in range(M)]  # 价值

        # n, m = map(int, input().split())
        n, m = 1500, 7
        # n //= 10  # 价格为10的整数倍，节省时间

        for i in range(1, m + 1):
            for inp in ["500 1 0", "400 4 0", "300 5 1", "400 5 1", "200 5 0", "500 4 0", "400 4 0"]:
                x, y, z = map(int, inp.split())
                # x //= 10
                if z == 0:
                    for t in range(4):
                        v[i][t] = v[i][t] + x
                        w[i][t] = w[i][t] + x * y

                elif v[z][1] == v[z][0]:  # 如果a==b，添加附件1(如果a=b=c=d说明没有附件)
                    v[z][1], w[z][1] = v[z][1] + x, w[z][1] + x * y
                    v[z][3], w[z][3] = v[z][3] + x, w[z][3] + x * y

                else:  # 添加附件2
                    v[z][2], w[z][2] = v[z][2] + x, w[z][2] + x * y
                    v[z][3], w[z][3] = v[z][3] + x, w[z][3] + x * y

        for i in range(1, m + 1):
            for j in range(n, -1, -1):
                for k in range(4):
                    if j >= v[i][k]:
                        f[j] = max(f[j], f[j - v[i][k]] + w[i][k])
        print(9 * f[n])


def knapsack(items, n, w):
    """
    :param items: 物品重量列表如 [2,2,4,6,3]
    :param n: 物品个数
    :param w: 背包可承载的重量
    :return:
    """
    states = [[False for _ in range(w + 1)] for _ in range(n)]  # 默认值为 False
    states[0][0] = True  # 第一行的值要特殊处理，可以利用哨兵优化
    if items[0] <= w:
        states[0][items[0]] = True  # 开始第一次做选择

    for i in range(1, n):  # 动态规划状态转移
        for j in range(0, w + 1):  # 不把第 i 个物品放入背包
            if states[i - 1][j]:
                states[i][j] = True

        for j in range(0, w - items[i] + 1):  # 把第 i 个物品放入背包
            if states[i - 1][j]:
                states[i][j + items[i]] = True

    for i in range(w, -1, -1):  # 输出结果
        if states[n - 1][i]:
            return i
    return 0


def knapsack2(items, n, w):
    """
    :param items: 物品重量列表如: [2,2,4,6,3]
    :param n: 物品个数
    :param w: 背包可承载的重量 11
    :return:
    """
    states = [False for _ in range(w + 1)]  # 默认值为 False
    states[0] = True  # 第一行的值要特殊处理，可以利用哨兵优化
    if items[0] <= w:
        states[items[0]] = True

    for i in range(1, n):  # 动态规划
        for j in range(w - items[i], -1, -1):  # 把第 i 个物品放入背包，j 需要从大到小来处理。
            if states[j]:  # 如果我们按照 j 从小到大处理的话，会出现 for 循环重复计算的问题
                states[j + items[i]] = True

    for i in range(w, -1, -1):  # 输出结果
        if states[i]:
            return i
    return 0


def knapsack3(items, values, n, w):
    """
    :param items: 物品重量列表如: [2,2,4,6,3]
    :param values: 物品价值列表: [3,3,4,2,1]
    :param n: 物品个数 5
    :param w: 背包可承载的重量 9
    :return:
    """
    states = [[None for _ in range(w + 1)] for _ in range(n)]  # 默认值为 None

    # 初始化 states
    for i in range(1, n):
        for j in range(0, w + 1):
            states[i][j] = -1

    states[0][0] = 0
    if items[0] <= w:
        states[0][items[0]] = values[0]  # 初始化第一行

    for i in range(1, n):  # 动态规划，状态转移
        for j in range(0, w + 1):  # 不选择第 i 个物品
            if states[i - 1][j] >= 0:
                states[i][j] = states[i - 1][j]

        for j in range(0, w - items[i] + 1):  # 选择第 i 个物品
            if states[i - 1][j] >= 0:
                v = states[i - 1][j] + values[i]
                if v > states[i][j + items[i]]:
                    states[i][j + items[i]] = v

    # 找最大值
    max_value = -1
    for j in range(0, w + 1):
        if states[n - 1][j] > max_value:
            max_value = states[n - 1][j]
    return max_value


def double11advance(items, n, w):
    """
    :param items:商品价格列表
    :param n: 商品个数
    :param w: 表示满减条件，比如 200
    :return:
    """
    states = [[False for _ in range(3 * w + 1)] for _ in range(n)]  # 默认值为 None
    states[0][0] = True  # 第一行的数据要特殊处理
    if items[0] <= 3 * w:
        states[0][items[0]] = True

    for i in range(1, n):  # 动态规划
        for j in range(0, 3 * w + 1):  # 不购买第 i 个商品
            if states[i - 1][j]:
                states[i][j] = states[i - 1][j]
        for j in range(0, 3 * w - items[i] + 1):  # 购买第 i 个商品
            if states[i - 1][j]:
                states[i][j + items[i]] = True

    j = None
    for j in range(w, 3 * w + 1):
        if states[n - 1][j]:  # 输出结果大于等于 w 的最小值
            break

    if j >= 3 * w:  # 没有可行性解
        return

    for i in range(n - 1, 0, -1):
        if j - items[i] >= 0 and states[i - 1][j - items[i]]:  # i 表示二维数组中的行，j 表示列
            print(items[i], ":buy")  # 购买这个商品
            j = j - items[i]
        else:
            print("not buy")  # 没有购买这个商品 j_index 没有变
    if j != 0:
        print(items[0], ":buy")  # 购买这个商品


# min_dist = sys.maxint


def min_dist_bt(i, j, dist, w, n):
    """
    :param i: 行下标
    :param j: 列下标
    :param dist: 当前的距离
    :param w: 一个二维数组，记录每个格子的距离值
    :param n: 一个多少个格子
    :return: 调用方式：min_dist_bt(0, 0, 0, [[1,3,5,9],[2,1,3,4],[5,2,6,7],[6,8,4,3]], 3)
    """
    global res
    if i == n and j == n:  # 到到 n-1位置了
        current_dist = dist + w[i][j]  # 到达指定位置时的路径数据之和，这里加上了指定位置自己的值
        if current_dist < res:
            res = current_dist
            print(res)
        return
    if i < n:  # 往下走，更新 i = i+1, j = j
        min_dist_bt(i + 1, j, dist + w[i][j], w, n)
    if j < n:  # 往右走，更新 i = i, j = j + 1
        min_dist_bt(i, j + 1, dist + w[i][j], w, n)


def min_dist_dp(matrix, n):
    """
    :param matrix: 一个二维数组，记录每个格子的距离值
    :param n: 一个多少个格子
    :return: 调用方式：min_dist_dp([[1,3,5,9],[2,1,3,4],[5,2,6,7],[6,8,4,3]], 3)
    """
    states = [[0 for _ in range(n)] for _ in range(n)]
    sum_dist = 0
    for j in range(0, n):  # 初始化 states 的第一行数据
        sum_dist += matrix[0][j]
        states[0][j] = sum_dist

    sum_dist = 0
    for i in range(0, n):  # 初始化 states 的第一列数据
        sum_dist += matrix[i][0]
        states[i][0] = sum_dist

    for i in range(1, n):
        for j in range(1, n):
            states[i][j] = matrix[i][j] + min(states[i][j - 1], states[i - 1][j])
    return states[n - 1][n - 1]


# matrix = [[1, 3, 5, 9], [2, 1, 3, 4], [5, 2, 6, 7], [6, 8, 4, 3]]
# n = 4
# mem = states = [[0 for _ in range(4)] for _ in range(4)]


def min_dist_zt(i, j):
    """
    :param i: 行下标
    :param j: 列下标
    :return 调用方式：min_dist_zt(n-1, n-1)
    """
    # global matrix, n, mem

    if i == 0 and j == 0:
        return matrix[0][0]

    if mem[i][j] > 0:
        return mem[i][j]

    min_left = sys.maxint
    if j - 1 >= 0:
        min_left = min_dist_zt(i, j - 1)

    min_right = sys.maxint
    if i - 1 >= 0:
        min_right = min_dist_zt(i - 1, j)

    current_min_dist = matrix[i][j] + min(min_left, min_right)
    mem[i][j] = current_min_dist
    return current_min_dist


def lwst_bt(i, j, e_dist, a_str, b_str, min_dist):
    """
    :param i: 字符串 a_str 的下标
    :param j: 字符串 b_str 的下标
    :param e_dist: 当前的编辑距离
    :param a_str: 字符串
    :param b_str: 字符串
    :param min_dist: 记录最小的编辑距离
    :return: 回朔法计算两个字符串的编辑距离 如：lwst_bt(0,0,0,"mitcmu","mtacnu",sys.maxint)
    """
    a_str_l = len(a_str)
    b_str_l = len(b_str)
    if i == a_str_l or j == b_str_l:
        if i < a_str_l:
            e_dist += a_str_l - i
        if j < b_str_l:
            e_dist += b_str_l - j
        if e_dist < min_dist:
            min_dist = e_dist
        print(min_dist)
        return
    if a_str[i] == b_str[j]:  # 两个字符匹配
        lwst_bt(i + 1, j + 1, e_dist, a_str, b_str, min_dist)
    else:  # 两个字符不匹配
        lwst_bt(i + 1, j, e_dist + 1, a_str, b_str, min_dist)  # 删除 a_str[i] 或者在 b_str[j] 前面添加一个跟 a_str[i] 相同的字符
        lwst_bt(i, j + 1, e_dist + 1, a_str, b_str, min_dist)  # 删除 b_str[j] 或者在 a_str[i] 前面添加一个跟 b_str[j] 相同的字符
        lwst_bt(i + 1, j + 1, e_dist + 1, a_str, b_str, min_dist)  # 将 a_str[i] 和 b_str[j] 替换为相同字符


def lcs(a_str, b_str):
    """ 计算编辑距离
        :param a_str: 字符串
        :param b_str: 字符串
        :return:
        """
    m = len(a_str)
    n = len(b_str)

    edit = [[i + j for j in range(n+1)] for i in range(m+1)]
    for x in range(1, m+1):
        for y in range(1, n+1):
            d = 0 if a_str[x-1] == b_str[y-1] else 1
            edit[x][y] = min(edit[x - 1][y] + 1, edit[x][y - 1] + 1, edit[x - 1][y - 1] + d)
    return edit[m][n]


class Graph:
    def __init__(self, v):
        self.v = v  # 顶点个数
        self.adj = [None] * v  # 邻接表
        for i in range(0, v):
            self.adj[i] = []

    def add_edge(self, s, t):  # s 先于 t, 边 s -> t
        self.adj[s].append(t)

    def topo_sort_by_kahn(self):
        in_degree = [0] * self.v  # 统计每个顶点的入度
        for i in range(0, self.v):
            for j in range(0, len(self.adj[i])):
                w = self.adj[i][j]
                in_degree[w] += 1

        queue = deque()
        for i in range(0, self.v):
            if in_degree[i] == 0:
                queue.append(i)
        while len(queue) > 0:
            i = queue.pop()
            print("->", i)
            for j in range(0, len(self.adj[i])):
                k = self.adj[i][j]
                in_degree[k] -= 1
                if in_degree[k] == 0:
                    queue.append(k)

    def topo_sort_by_dfs(self):
        inverse_adj = [None] * self.v
        for i in range(0, self.v):
            inverse_adj[i] = []
        for i in range(0, self.v):
            for j in range(0, len(self.adj[i])):
                w = self.adj[i][j]
                inverse_adj[w].append(i)

        visited = [False] * self.v
        for i in range(0, self.v):
            if not visited[i]:
                visited[i] = True
                self.dfs(i, inverse_adj, visited)

    def dfs(self, vertex, inverse_adj, visited):
        for i in range(0, len(inverse_adj[vertex])):
            w = inverse_adj[vertex][i]
            if visited[w]:
                continue
            visited[w] = True
            self.dfs(w, inverse_adj, visited)
        print("->", vertex)


class Dijkstra:
    def __init__(self):
        self.graph = {}  # 记录地图指向关系，和相应的权重
        self.costs = {}  # 记录从起点到每个点的距离
        self.parents = {}  # 记录每个点的父节点
        self.processed = []  # 记录以及处理里的节点

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


class AStar:
    def __init__(self):
        self.graph = {}  # 记录地图指向关系，和相应的权重
        self.costs = {}  # 记录从起点到每个点的距离
        self.parents = {}  # 记录每个点的父节点
        self.processed = []  # 记录以及处理里的节点

    def add_edge(self, s, t, w, f):  # s 先于 t, 边 s -> t
        if not self.graph.get(s):
            self.graph[s] = {}
        if t and f:
            self.graph[s][t]["w"] = w
            self.graph[s][t]["f"] = f

    def dijkstra(self):
        node = self.find_lowest_f_node()  # 在未处理的节点中找出开销最小的节点
        while node is not None:  # while循环在所有节点都被处理过后结束
            cost = self.costs[node]
            neighbors = self.graph[node]
            for n in neighbors.keys():
                new_cost = cost + neighbors[n]
                if self.costs[n] > new_cost:  # 如果经当前节点前往该邻居更近
                    self.costs[n] = new_cost  # 更新该邻居节点的开销
                    self.parents[n] = node  # 该邻居节点的父节点设置为当前节点
            self.processed.append(node)  # 将当前节点标记为处理过
            node = self.find_lowest_f_node()  # 找出接下来要处理的节点，并循环
        print(self.costs)
        print(self.parents)

    def find_lowest_f_node(self):
        lowest_f = float("inf")
        lowest_f_node = None
        for node in self.costs:  # 遍历所有的节点
            f = self.costs[node]
            if f < lowest_f and node not in self.processed:
                lowest_f = f  # 就将其视为开销最低的节点
                lowest_f_node = node
        return lowest_f_node


class Bitmap:
    def __init__(self, max_value):
        """确定所需数组个数"""
        self.size = int((max_value + 31 - 1) / 31)  # 初始化bitmap 向上取整
        self.array = [0 for _ in range(self.size)]

    def set_1(self, num):
        """将元素所在的位置1"""
        elem_index = num / 31  # 计算在数组中的索引
        byte_index = num % 31  # 计算在数组中的位索引
        ele = self.array[elem_index]
        self.array[elem_index] = ele | (1 << byte_index)  # 相关位置1, 左移 byte_index 位相当于乘以2的 byte_index 次方

    def test_1(self, num):
        """检测元素存在的位置"""
        elem_index = num / 31
        byte_index = num % 31
        if self.array[elem_index] & (1 << byte_index):
            return True
        return False


def fn(a):
    return a % 2 == 1


class Singleton(object):
    __instance = None

    def __new__(cls, age, name):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance


class TimeException(Exception):
    def __init__(self, exception_info):
        super.__init__()
        self.info = exception_info

    def __str__(self):
        return self.info


def time_check(func):
    import datetime

    def wrapper(*args, **kwargs):
        if datetime.datetime.now().year == 2019:
            func(*args, **kwargs)
        else:
            raise TimeException("函数已过时")

    return wrapper


def find_2(a_list, target):
    low = 0
    high = len(a_list) - 1
    mid = (low + high) / 2
    while low <= high:
        if a_list[mid] == target:
            return mid
        if target < a_list[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n


def _not_divisible(n):
    return lambda x: x % n > 0


def primes():
    yield 2
    it = _odd_iter()  # 初始序列
    while True:
        n = next(it)  # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it)  # 构造新序列


def is_palindrome(n):
    _length = len(str(n))
    if _length % 2 == 1:
        return str(n)[0:_length / 2] == str(n)[_length / 2 + 1:][::-1]
    else:
        return str(n)[0:_length / 2] == str(n)[_length / 2:][::-1]


# def createCounter():
#     def seed():
#         s = 0
#         while True:
#             s += 1
#             yield s
#
#     _iter = seed()
#     def counter():
#         return next(_iter)
#     return counter

# def createCounter():
#
#     def counter1():
#         s = 0
#         while True:
#             s += 1
#             yield s
#
#     test = counter1()
#
#     def counter2():
#         return next(test)
#
#     return counter2
# import asyncio
# # from aiohttp import web
#
# async def index(request):
#     await asyncio.sleep(0.5)
#     return web.Response(body=b'<h1>Index</h1>')
#
# async def hello(request):
#     await asyncio.sleep(0.5)
#     text = '<h1>hello, %s!</h1>' % request.match_info['name']
#     return web.Response(body=text.encode('utf-8'))
#
# async def init(loop):
#     app = web.Application(loop=loop)
#     app.router.add_route('GET', '/', index)
#     app.router.add_route('GET', '/hello/{name}', hello)
#     srv = await loop.create_server(app.make_handler(), '127.0.0.1', 8000)
#     print('Server started at http://127.0.0.1:8000...')
#     return srv

# loop = asyncio.get_event_loop()
# loop.run_until_complete(init(loop))
# loop.run_forever()
import os
# 将字符串的时间转换为时间戳
import time


class SingleTong(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(SingleTong, cls).__new__(cls, *args, **kwargs)
        return cls._instance


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def log(text):
    def decorate(func):
        def wrapper(*args, **kwargs):
            print("%s fuck you" % text)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def bin_or_oct_to_dec(num, base):
    """二进制, 八进制, 转十进制, base =2,8"""
    result = 0
    length = len(num)
    for x in range(length):
        result += base ** x * int(num[length - x - 1])
    return result
    # return int(num,2)
    # return int(num,8)


def hex_to_dec(num):
    """16进制转 10进制 """
    base = [str(x) for x in range(10)] + [chr(x) for x in
                                          range(ord('A'), ord("A") + 6)]  # 前者把 0 ~ 9 转换成字符串存进列表 base 里，后者把 A ~ F 存进列表
    # 相当于 base = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    result = 0
    length = len(num)
    for x in range(length):
        result += 16 ** x * int(base.index(num[length - x - 1]))
    return result
    # return int(num,16)


def dec_to_bin_or_oct(num, base):
    """十进制转二进制 或八进制 base=2 或者 8"""
    l = []  # 创建一个空列表
    if num < 0:  # 是负数转换成整数
        return "-" + dec_to_bin_or_oct(abs(num), base)  # 如过是负数，先转换成正数
    while True:
        num, reminder = divmod(num, base)  # 短除法，对2求，分别得到除数 和 余数、这是 Python 的特有的一个内置方法，分别可以到商 及 余数
        l.append(str(reminder))  # 把获得的余数 存入字符串
        if num == 0:  # 对应了前面的话，当商为 0时，就结束啦
            return "".join(l[::-1])  # 对列表中的字符串进行逆序拼接，得到一个二进制字符串
            # return bin(num)
            # return oct(num)


def dec_to_hex(num):
    """十进制转十六进制（这个相对麻烦一点，因为，十六进制包含 A-F，大小写不敏感）"""
    base = [str(x) for x in range(10)] + [chr(x) for x in
                                          range(ord('A'), ord("A") + 6)]  # 前者把 0 ~ 9 转换成字符串存进列表 base 里，后者把 A ~ F 存进列表
    # 相当于 base = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    l = []
    if num < 0:
        return "-" + dec_to_hex(abs(num))
    while True:
        num, rem = divmod(num, 16)  # 求商 和 留余数
        l.append(base[rem])
        if num == 0:
            return "".join(l[::-1])
            # return hex(num)


import sys

import sys


def ip2bin(mask):
    """
    ip地址转换为二进制数
    :param mask:
    :return: string类型
    """
    l = list()
    for i in mask:
        _ip = bin(i)[2:]
        if len(str(_ip)) < 8:
            _ip = "0" * (8 - len(str(_ip))) + str(_ip)
        l.extend(_ip)
    return ''.join(str(j) for j in l)


from collections import OrderedDict

test = OrderedDict()


def check_mask(mask):
    """

    :param mask: 存储mask四个十进制位的列表
    :return: mask合法返回True
    """
    if mask in ([255, 255, 255, 255], [0, 0, 0, 0]):
        return False
    b_ip = ip2bin(mask)
    flag = False  # 是否碰到了0
    for i, c in enumerate(b_ip):
        if flag:
            if c == '1':
                return False
        else:
            if c == '0':
                flag = True
    else:
        return True


def check_ip(ip):
    """

    :param ip: 存储ip四个十进制位的列表
    :return: ip合法返回True
    """
    for i in ip:
        try:
            temp = int(i)
        except:
            return False
        if temp < 0 or temp > 255:
            return False
    else:
        return True


def longestPalindrome(s):
    if s == s[::-1]:
        return len(s)

    maxLen = 0

    for i in range(len(s)):
        a_index = i - maxLen
        if a_index >= 1 and s[a_index - 1:i + 1] == s[a_index - 1:i + 1][::-1]:
            maxLen += 2
            continue
        if a_index >= 0 and s[a_index:i + 1] == s[a_index:i + 1][::-1]:
            maxLen += 1

    return maxLen


def get_all_rabits(n):
    if n in [1, 2]:
        return 1
    if n == 3:
        return 2
    else:
        res = get_all_rabits(n - 1) + 2 * get_all_rabits(n - 4)
        return res


def check_ip_is_ok(ip):
    try:
        ip_arr = str(ip).split(".")
        for i in ip_arr:
            if int(i) < 0 or int(i) > 255:
                return False
        return True
    except:
        return False


def check_mask_is_ok(mask):
    try:
        mask_arr = str(mask).split(".")
        for i in mask_arr:
            bin_str = bin(i).replace("0b", "")
            if len(bin_str) < 8 and "1" in str(bin_str):
                return False
        return True
    except:
        return False


class Node(object):
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    def __init__(self, node=None):
        self._head = node

    def length(self):
        cur = self._head
        count = 0
        while cur:
            cur = cur.next
            count += 1
        return count

    def travel(self):
        cur = self._head
        while cur:
            print(cur.elem, end=" ")
            cur = cur.next
        print("")

    def add(self, item):
        node = Node(item)
        node.next = self._head
        self._head = node

    def insert(self, item, index):
        node = Node(item)
        if not self._head:
            self._head = node
        else:
            cur = self._head
            while cur:
                if cur.elem == index:
                    node.next = cur.next
                    cur.next = node
                    break
                else:
                    cur = cur.next

    def remove(self, item):
        cur = self._head
        pre = None
        while cur:
            if cur.elem == item:
                if cur == self._head:
                    self._head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

            class Node(object):
                def __init__(self, data):
                    self.data = data
                    self.next = None

            class SingleLinkList(object):
                def __init__(self, node=None):
                    self._head = node

                def add(self, data):
                    node = Node(data)
                    node.next = self._head
                    self._head = node

                def length(self):
                    cur = self._head
                    count = 0
                    while cur:
                        count += 1
                        cur = cur.next
                    return count

                def traval(self):
                    cur = self._head
                    while cur:
                        print(cur.data, end=" ")
                        cur = cur.next
                    print("")

                def insert(self, data, target_data):
                    node = Node(data)
                    if not self._head:
                        self._head = node
                    else:
                        cur = self._head
                        while cur:
                            if cur.data == target_data:
                                node.next = cur.next
                                cur.next = node
                                break
                            else:
                                cur = cur.next

                def remove(self, target_data):
                    cur = self._head
                    pre = None
                    while cur.next:
                        if cur.data == target_data:
                            if cur == self._head:
                                self.head = cur.next
                            else:
                                pre.next = cur.next
                            break
                        else:
                            pre = cur
                            cur = cur.next

                a = sys.stdin.readlines().split(" ")
                a = list(map(int, a))
                node_num = a[0]
                head_node = Node(a[1])
                nodes = a[2:len(a) - 1]
                to_del_data = a[-1]
                single_link_list = SingleLinkList(head_node)
                for i in range(0, len(nodes), 2):
                    data = nodes[i]
                    target_data = nodes[i + 1]
                    single_link_list.insert(data, target_data)

                single_link_list.remove(to_del_data)
                single_link_list.traval()


# from threading import Thread, Condition
#
#
# res = []
# condition = Condition()
# current = "A"


def send_a(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "A":
                condition.wait()
            res.append("A")
            current = "B"
            condition.notify_all()


def send_b(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "B":
                condition.wait()
            res.append("B")
            current = "C"
            condition.notify_all()


def send_c(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "C":
                condition.wait()
            res.append("C")
            current = "D"
            condition.notify_all()


def send_d(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "D":
                condition.wait()
            res.append("D")
            current = "A"
            condition.notify_all()


#
# if __name__ == '__main__':
#     a = 1453
#     thread_list = [Thread(target=send_a, args=(a,)), Thread(target=send_b, args=(a,)), Thread(target=send_c, args=(a,)),
#                    Thread(target=send_d, args=(a,))]
#     for thread in thread_list:
#         thread.start()
#
#     for thread in thread_list:
#         thread.join()
#     print("".join(res))
#
# import re


def calculate(n1, n2, operator):
    if operator == "+":
        result = n1 + n2
    elif operator == "-":
        result = n1 - n2
    elif operator == "*":
        result = n1 * n2
    elif operator == "/":
        result = n1 / n2
    else:
        raise Exception("operator is not support now...")
    return result


def is_operator(e):
    operators = ["+", "-", "*", "/", "(", ")"]
    return True if e in operators else False


def formula_format(formula):
    """
    步骤需要处理的是区分横杠‘-’是代表负数还是减号
    """

    formula = re.sub(' ', '', formula)

    # 以 '横杠数字' 分割， -> \- 表示匹配横杠开头； \d+ 表示匹配数字1次或多次；\.?表示匹配小数点0次或1次;\d*表示匹配数字1次或多次。
    formula_list = [i for i in re.split('(\-\d\.?\d*)', formula) if i]

    final_formula = []
    for item in formula_list:

        # 第一个是以横杠开头的数字（包括小数）final_formula。即第一个是负数，横杠就不是减号
        if len(final_formula) == 0 and re.search('^\-\d+\.?\d*$', item):
            final_formula.append(item)
            continue

        if len(final_formula) > 0:

            # 如果final_formal最后一个元素是运算符['+', '-', '*', '/', '('], 则横杠数字不是负数
            if re.search('[\+\-\*\/\(]$', final_formula[-1]):
                final_formula.append(item)
                continue

        # 剩下的按照运算符分割开
        item_split = [i for i in re.split('([\+\-\*\/\(\)])', item) if i]
        final_formula += item_split

    return final_formula


def decision(tail_op, now_op):
    """
    :param tail_op: 运算符栈最后一个运算符
    :param now_op: 算式列表取出当前运算符
    :return: 1 弹栈， 0 弹出运算符栈最后一个元素， -1 入栈
    """
    # 运算符等级
    rate1 = ["+", "-"]
    rate2 = ["*", "/"]
    rate3 = ["("]
    rate4 = [")"]

    if tail_op in rate1:
        if now_op in rate2 or now_op in rate3:
            # 运算符优先级不同
            return -1  # 把当前取出的运算符压栈 "1+2+3"
        else:
            return 1  # 否则运算符栈中最后的 运算符弹出，进行计算

    elif tail_op in rate2:
        if now_op in rate3:
            return -1
        else:
            return 1

    elif tail_op in rate3:
        if now_op in rate4:
            return 0  # ( 遇上 ) 需要弹出 (，丢掉 )
        else:
            return -1  # 只要栈顶元素为(，当前元素不是)都应入栈
    else:
        return -1


# 主函数  ->  遍历算式列表中的字符，决定入栈弹栈操作
def final_cal(formula_list):
    num_stack = []
    op_stack = []
    for e in formula_list:
        operator = is_operator(e)
        if not operator:
            # 数字栈 ->  转为浮点数
            a = 2
            num_stack.append(float(int(e)))
        else:
            # e 开始进入 运算符栈, -> 判断
            while True:
                a = 1
                if len(op_stack) == 0:  # 第一个运算符来了，都得入栈
                    op_stack.append(e)
                    break
                # 后面运算符来了，需要判断入栈，or 出栈。
                pop_oper = op_stack[-1]
                tag = decision(op_stack[-1], e)
                if tag == -1:  # 压栈
                    op_stack.append(e)
                    break
                elif tag == 0:  # 弹出运算符栈内最后一个 "("， 丢掉当前的 ")", 进入下次循环
                    op_stack.pop()
                    break
                elif tag == 1:  # 运算符栈弹出最后一个运算符，数字栈弹出最后两个元素，进行计算
                    op = op_stack.pop()
                    num2 = num_stack.pop()
                    num1 = num_stack.pop()

                    # 计算后结果 --> 压入数字栈
                    num_stack.append(calculate(num1, num2, op))

    # 处理大循环结束后 数字栈和运算符栈中可能还有元素 的情况
    while len(op_stack) != 0:
        op = op_stack.pop()
        num2 = num_stack.pop()
        num1 = num_stack.pop()
        num_stack.append(calculate(num1, num2, op))
    return num_stack, op_stack


class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None


class SingleLinkedList(object):
    def __init__(self):
        self._head = None
        self.count = 0

    def add(self, node):
        if not self._head:
            self._head = node
        else:
            node.next = self._head
            self._head = node
        self.count += 1

    def get_node_by_index(self, index):
        if index > self.count:
            print(None)
        else:
            i = 0
            cur = self._head
            while i < self.count - index:
                cur = cur.next
                i += 1
            print(cur.data)



def config_file_recover(x):
    """配置文件恢复算法"""
    while True:
        try:
            commands_dic = {"reset": "reset what", "reset board": "board fault", "board add": "where to add",
                            "board delete": "no board at all", "reboot backplane": "impossible",
                            "backplane abort": "install first", "he he": "unknown command"}

            commands = input().rstrip("\n").split(" ")
            if len(commands) == 1:
                flag = False
                for k, v in commands_dic.items():
                    if len(k.split(" ")) == 1 and k.startswith("".join(commands)):
                        print(v)
                        flag = True
                        break
                if not flag:
                    print("unknown command")

            else:
                first_command_matched = []
                for k, v in commands_dic.items():
                    if k.startswith(commands[0]):
                        first_command_matched.append(k)
                first_command_matched = [x for x in first_command_matched if
                                         len(x.split(" ")) > 1 and x.split(" ")[1].startswith(commands[1])]
                if len(first_command_matched) == 1:
                    print(commands_dic.get(first_command_matched[0]))
                else:
                    print("unknown command")
        except Exception as e:
            print(e)
            break


# 题目的隐藏条件好像是，不考虑使用括号，数字位置可调
def helper(arr, item):
    if item < 1:
        return False
    if len(arr) == 1:
        if arr[0] == item:
            print(123)
        return arr[0] == item

    for i in range(len(arr)):
        L = arr[:i] + arr[i+1:]
        v = arr[i]
        if helper(L, item-v) or helper(L, item+v) or helper(L, item*v) or helper(L, item/v):
            return True
    return False



import itertools

def helper1(nums):
    formats = ['(({0[0]}{1[0]}{0[1]}){1[1]}{0[2]}){1[2]}{0[3]}',
               '({0[0]}{1[0]}{0[1]}){1[1]}({0[2]}{1[2]}{0[3]})',
               '({0[0]}{1[0]}({0[1]}{1[1]}{0[2]})){1[2]}{0[3]}',
               '{0[0]}{1[0]}(({0[1]}{1[1]}{0[2]}){1[2]}{0[3]})',
               '{0[0]}{1[0]}({0[1]}{1[1]}({0[2]}{1[2]}{0[3]}))']
    operators = '+-*/'
    breakFlag = False
    for num in itertools.permutations(nums, 4):  # 返回所有数字的排列方式A44
        # 返回所有运算符的可能4^3
        for operator in itertools.product(operators, repeat=3):
            for f in formats:
                exp = f.format(num, operator)
                try:
                    res = eval(exp)
                    if res == 24:
                        # print(exp + '=24')
                        print('true')
                        breakFlag = True
                        break
                except ZeroDivisionError:
                    continue
            if breakFlag:
                break
        if breakFlag:
            break
    if not breakFlag:
        print('false')


if __name__ == "__main__":
    # input_data_permutation = itertools.permutations(range(4), 4)
    for x in itertools.permutations(range(4), 4):
        print(x)


































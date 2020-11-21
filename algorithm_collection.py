# -*- coding: utf-8 -*-


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


class Heap(object):
    """大顶堆的定义，实现插入和删除"""

    def __init__(self, capacity):
        self.a = [None] * (capacity + 1)
        self.n = capacity
        self.count = 0

    def insert(self, data):
        """插入一个数据"""
        if self.count >= self.n:
            return False
        self.count += 1
        i = self.count
        self.a[i] = data
        # 让新插入的节点与父节点对比大小。如果不满足子节点小于等于父节点的大小关系，
        # 我们就互换两个节点。一直重复这个过程，直到所有的子节点小于等于父节点。
        while int(i / 2) > 0 and self.a[i] > self.a[int(i / 2)]:
            temp = self.a[i]
            self.a[i] = self.a[int(i / 2)]
            self.a[int(i / 2)] = temp
            i = int(i / 2)
        print(self.a)
        return True

    def remove_max(self):
        """删除大顶堆最大值"""
        if self.count == 0:
            return False
        self.a[1] = self.a[self.count]
        self.count -= 1
        self.heapify()
        print(self.a)
        return True

    def heapify(self):
        # 删除堆顶元素之后，就需要把第二大的元素放到堆顶，那第二大元素肯定会出现在左右子节点中。
        # 然后我们再迭代地删除第二大节点，以此类推，直到叶子节点被删除。
        i = 1
        while True:
            max_pos = i
            if 2 * i <= self.count and self.a[i] < self.a[2 * i]:
                max_pos = 2 * i
            if 2 * i + 1 <= self.count and self.a[max_pos] < self.a[2 * i + 1]:
                max_pos = 2 * i + 1
            if max_pos == i:
                self.a[self.count + 1] = None
                break
            temp = self.a[i]
            self.a[i] = self.a[max_pos]
            self.a[max_pos] = temp
            i = max_pos

    def build_heap(self, arr):
        """算法从第1个下标开始计算，所以需要插入站位None"""
        n = len(arr)
        arr.insert(0, None)
        # 我们只对 n/2 开始到 1的数据进行堆化，下标是n/2+ 1 到 n 的节点都是叶子节点，不需要堆化，
        # 对于完全二叉树来说，下标是n/2+ 1 到 n 的节点都是叶子节点
        for i in range(int(n / 2), 0, -1):
            self.heapify_by_arr(arr, n, i)
        return arr

    @staticmethod
    def heapify_by_arr(arr, l, i):
        while True:
            max_pos = i
            if 2 * i <= l and arr[i] < arr[2 * i]:
                max_pos = 2 * i
            if 2 * i + 1 <= l and arr[max_pos] < arr[2 * i + 1]:
                max_pos = 2 * i + 1
            if max_pos == i:
                break
            temp = arr[i]
            arr[i] = arr[max_pos]
            arr[max_pos] = temp
            i = max_pos

    def sort_by_heap(self, arr):
        """利用堆进行数组排序"""
        k = len(arr)
        arr = self.build_heap(arr)
        while k > 1:
            temp = arr[k]
            arr[k] = arr[1]
            arr[1] = temp
            k -= 1
            self.heapify_by_arr(arr, k, 1)
        print(arr)


class Graph(object):
    """图的广度优先搜索和深度优先搜索"""

    def __init__(self, nodes, sides):
        """
        nodes 表示点，如：nodes = [i for i in range(8)]
        sides 表示边，如：sides = [(0, 1),(0, 3),(1, 2),(1, 4),(1, 0),(2, 1),(2, 5),(3, 0),(3, 4),(4, 1),(4, 3),(4, 5),(4, 6),(5, 2),(5, 4),(5, 7),(6, 4),(6, 7),(7, 6),(7, 5)]
        """
        self.sequence = {}  # self.sequence，key是点，value是与key相连接的点
        self.side = []  # self.side是临时变量，主要用于保存与指定点相连接的点
        for node in nodes:
            for side in sides:
                u, v = side
                # 指定点与另一个点在同一个边中，则说明这个点与指定点是相连接的点，则需要将这个点放到self.side中
                if node == u and v not in self.side:
                    self.side.append(v)
                elif node == v and u not in self.side:
                    self.side.append(u)
            self.sequence[node] = self.side
            self.side = []
            print(self.sequence)

    def bfs(self, s, t):
        """
        广度优先算发
        是指定节点 s 开始，沿着树的宽度遍历树的节点。如果所有节点均被访问，则算法中止。
        广度优先搜索的实现一般采用open-closed表。
        s 表示开始节点，t 表示要找的节点
        """

        queue, order = [], []  # queue本质上是堆栈，用来存放需要进行遍历的数据, order里面存放的是具体的访问路径
        queue.append(s)  # 首先将初始遍历的节点放到queue中，表示将要从这个点开始遍历
        order.append(s)  # 由于是广度优先，也就是先访问初始节点的所有的子节点
        while queue:
            # queue.pop(0)意味着是队列的方式出元素，就是先进先出，而下面的for循环将节点v的所有子节点
            # 放到queue中，所以queue.pop(0)就实现了每次访问都是先将元素的子节点访问完毕，而不是优先叶子节点
            v = queue.pop(0)
            for w in self.sequence[v]:
                if w not in order and w not in queue:
                    # 这里可以直接order.append(w) 因为广度优先就是先访问节点的所有下级子节点，所以可以
                    # 将self.sequence[v]的值直接全部先给到order
                    order.append(w)
                    queue.append(w)
                if w == t:
                    print("Got it")
                    return order
        return order

    def dfs(self, s, t):
        """
        深度优先算法，是一种用于遍历或搜索树或图的算法。沿着树的深度遍历树的节点，尽可能深的搜索树的分支。
        当节点v的所在边都己被探寻过，搜索将回溯到发现节点v的那条边的起始节点。
        这一过程一直进行到已发现从源节点可达的所有节点为止。如果还存在未被发现的节点，
        则选择其中一个作为源节点并重复以上过程，整个进程反复进行直到所有节点都被访问为止。属于盲目搜索。
        s 表示开始节点，t 表示要找的节点
        """

        queue, order = [], []  # queue本质上是堆栈，用来存放需要进行遍历的数据, order里面存放的是具体的访问路径
        queue.append(s)  # 首先将初始遍历的节点放到queue中，表示将要从这个点开始遍历
        while queue:
            # 从queue中pop出点v，然后从v点开始遍历了，所以可以将这个点pop出，然后将其放入order中
            # 这里才是最有用的地方，pop（）表示弹出栈顶，由于下面的for循环不断的访问子节点，并将子节点压入堆栈，
            # 也就保证了每次的栈顶弹出的顺序是下面的节点
            v = queue.pop()
            order.append(v)
            # 这里开始遍历v的子节点
            for w in self.sequence[v]:
                # w既不属于queue也不属于order，意味着这个点没被访问过，所以讲起放到queue中，然后后续进行访问
                if w not in order and w not in queue:
                    queue.append(w)
                if w == t:
                    print("Got it")
                    order.append(w)
                    return order
        return order


class StringMatching(object):
    """常见字符串匹配算法"""

    @staticmethod
    def bf(main_str, sub_str):
        """
        BF 是 Brute Force 的缩写，中文叫作暴力匹配算法
        在主串中，检查起始位置分别是 0、1、2…n-m 且长度为 m 的 n-m+1 个子串，看有没有跟模式串匹配的
        """
        a = len(main_str)
        b = len(sub_str)
        for i in range(a - b + 1):
            if main_str[i:i + b] == sub_str:
                return i
        return -1

    @staticmethod
    def generate_bc(sub_str):
        """
        :param sub_str: 是模式串
        :return: 返回一个散列表  散列表的下标是 b_char 中每个字符的 ascii 码值，下标对应的值是该字符在b_char中最后一次出现的位置
        """
        ascii_size = 256
        ascii_list = [-1] * ascii_size  # 初始化一个散列表

        for i in range(0, len(sub_str)):
            ascii_val = ord(sub_str[i])  # 计算 b_char中每个字符的 ascii 值
            ascii_list[ascii_val] = i  # 存每个字符在 b_char 最后一次出现的位置
        return ascii_list

    @staticmethod
    def move_by_gs(j, m, suffix, prefix):
        """
        :param j: 表示坏字符对应的模式串中的字符下标
        :param m: m 表示模式串的长度
        :param suffix:  suffix 数组中下标k，表示后缀子串的长度，下标对应的数组值存储的是后缀子串在字符串sub_str的位置
        :param prefix: 而 prefix 数组中下标k，表示后缀子串的长度，对应的值 True 或者 False 则表示该后缀子串跟相同长度的前缀子串是否相对
        :return: 应该往后滑动的距离
        """
        k = m - 1 - j  # k 好后缀长度
        if suffix[k] != -1:  # 如果后缀在字符串中的所有前缀子串中存在
            return j - suffix[k] + 1
        for r in range(j + 2, m):  # 如果有后缀等于前缀
            if prefix[m - r]:
                return r
        return m  # 如果前面两个规则都不使用，则直接往后滑动 m 位

    @staticmethod
    def generate_gs(sub_str):
        """
        假如字符串sub_str = cabcab, 那么后缀子串有[b,ab,cab,bcab,abcab]
        suffix 数组中下标k，表示后缀子串的长度，下标对应的数组值存储的是后缀子串在字符串sub_str的位置
        如：suffix[1] = 2, suffix[2] = 1, suffix[3] = 0, suffix[4] = -1, suffix[5] = -1

        而 prefix 数组中下标k，表示后缀子串的长度，对应的值 True 或者 False 则表示该后缀子串跟相同长度的前缀子串是否相对
        如 prefix[1] = False, prefix[2] = False, prefix[3] = True （表示 后缀cba = 前缀cba）, prefix[4] = False, prefix[5] = False,
        """
        m = len(sub_str)
        suffix = [-1] * m
        prefix = [False] * m
        for i in range(0, m - 1):
            j = i
            k = 0  # 公共后缀串长度，
            while j >= 0 and sub_str[j] == sub_str[m - 1 - k]:  # 求公共后缀子串，先计算第一个跟模式子串最后一个字符相匹配的位置
                j -= 1  # 然后依次比较前面的字符是否相等，比如：cabcab，先从前往后历遍，计算得到 b 字符是符合要求的，
                k += 1  # 此时再从 b 字符前一个位置与模式串倒数第二个位置的字符去比较，如果还是相等，则继续，循环了 k 次说明匹配的字符长度就是 k
                suffix[k] = j + 1  # j+1 表示公共后缀子串在 sub_str[0:i]中的起始下标

            if j == -1:  # 如果公共后缀子串也是模式串的前缀子串
                prefix[k] = True
        return suffix, prefix

    def bm_simple(self, main_str, sub_str):
        """
        :param main_str: 主字符串
        :param sub_str: 模式串
        :return:
        仅用坏字符规则，并且不考虑 si-xi 计算得到的移动位数可能会出现负数的情况的代码实现如下
        """
        bc = self.generate_bc(sub_str)  # 记录模式串中每个字符最后出现的位置，也就是坏字符的哈希表
        i = 0  # 表示主串和模式串对齐的第一个字符
        n = len(main_str)  # 表示主串的长度
        m = len(sub_str)  # 表示子串的长度
        while i <= n - m:  # i 最多滑动到两个字符右对齐的位置
            j = m - 1  # 数组下标从0开始算，这里实际从子字符串的最后一个位置开始找坏字符
            while j >= 0:
                if main_str[i + j] != sub_str[j]:  # 坏字符对应模式串中的下标是 j
                    break
                j -= 1
            if j < 0:  # 没有坏字符，全部都匹配上了
                return i  # 匹配成功，返回主串和模式串第一个匹配的字符的位置
            bad_str_index = bc[ord(main_str[i + j])]  # 坏字符在子模式串中的位置
            i += (j - bad_str_index)  # 这里等同于将模式串往后滑动  j - bc[ord(main_str[i + j]) 位
        return -1

    def bm(self, main_str, sub_str):
        """
        :param main_str: 主字符串
        :param sub_str: 模式串
        :return:
        """
        bc = self.generate_bc(sub_str)  # 记录模式串中每个字符最后出现的位置，也就是坏字符的哈希表
        suffix, prefix = self.generate_gs(sub_str)

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
                y = self.move_by_gs(j, m, suffix, prefix)  # 计算好后缀规则每次需要往后滑动的次数
            i = i + max(x, y)

        return -1

    @staticmethod
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

    def kmp(self, main_str, sub_str):
        n = len(main_str)
        m = len(sub_str)
        next_list = self.get_next_list(sub_str)
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


if __name__ == "__main__":
    s = StringMatching()
    print(s.kmp("dferrcbacbagrgrgk", "cbacba"))

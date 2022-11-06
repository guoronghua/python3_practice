def fun_01():
    """
    勾股数元组（ 如果3个正整数(a,b,c)满足a**2 + b**2 = c**2的关系）
    如果3个正整数(a,b,c)满足a**2 + b**2 = c**2的关系，则称(a,b,c)为勾股数（著名的勾三股四弦
    五），为了探索勾股数的规律，我们定义如果勾股数(a,b,c)之间两两互质（即a与b，a与c，b与
    c之间均互质，没有公约数），则其为勾股数元祖（例如(3,4,5)是勾股数元祖，(6,8,10)则不
    是勾股数元祖）。请求出给定范围[N,M]内，所有的勾股数元祖。

    输入描述:
    1
    20

    输出
     3 4 5
     2 12 13
     8 15 17

    """

    def hu_zhi(a, b):
        if a == 0 or b == 0:
            return 1
        if a % b == 0:
            return b
        else:
            return hu_zhi(b, a % b)

    n = input()
    m = input()
    count = 0
    for i in range(n, m):
        for j in range(n + 1, m):
            for k in range(n + 2, m):
                if i < j < k and k * k == i * i + j * j and hu_zhi(i, j) == 1 and hu_zhi(j, k) == 1 and hu_zhi(i,
                                                                                                               k) == 1:
                    print("%s %s %s" % (i, j, k))
                    count += 1


def fun_02():
    """
    给定两个整数数组array1、array2，数组元素按升序排列。假设从array1、array2中分别取出一个元素可构成一对元素，现在需要取出k对元素，并对取出的所有元素求和，计算和的最小值
    注意：两对元素如果对应于array1、array2中的两个下标均相同，则视为同一对元素。

    输入描述:
    输入两行数组array1、array2，每行首个数字为数组大小size(0 < size <= 100);
    0 < array1[i] <= 1000
    0 < array2[i] <= 1000
    接下来一行为正整数k
    0 < k <= array1.size() * array2.size()
    输出描述:
    满足要求的最小和

    示例1
    输入
    3 1 1 2
    3 1 2 3
    2
    输出
    4
    说明
    用例中，需要取2对元素
    取第一个数组第0个元素与第二个数组第0个元素组成1对元素[1,1];
    取第一个数组第1个元素与第二个数组第0个元素组成1对元素[1,1];
    求和为1+1+1+1=4，为满足要求的最小和
    """
    list_01 = input()[1:]
    list_02 = input()[1:]
    need_cnt = input()
    res = []
    for x in list_01[:need_cnt]:
        for y in list_02[:need_cnt]:
            res.append(x + y)
    res.sort()
    return sum(res[:need_cnt])


def fun_03(n):
    """
    一只顽猴想要从山脚爬到山顶
    途中经过一个有n个台阶的阶梯，但是这个猴子有个习惯，每一次只跳1步或3步
    试问？猴子通过这个阶梯有多少种不同的跳跃方式
    """
    f1 = 1
    f2 = 1
    f3 = 2
    f4 = 1 if n in [1, 2] else 2
    for x in range(4, n + 1):
        f4 = f3 + f1
        f1 = f2
        f2 = f3
        f3 = f4
    return f4


def fun_04(nums):
    """
    找终点 给定一个正整数数组
    求从第一个成员开始正好走到数组最后一个成员所使用的最小步骤数
    1. 第一步 必须从第一元素起 且 1<=第一步步长<len/2 (len为数组长度)
    2. 从第二步开始只能以所在成员的数字走相应的步数，不能多不能少，
    如果目标不可达返回-1
    只输出最小的步骤数量
    """
    step_list = []

    def do_check(i):
        res = i
        _step = 1
        while True:
            res += nums[i]
            _step += 1
            if res == len(nums) - 1:
                return _step
            if res < len(nums) - 1:
                i = res
            else:
                return -1

    for x in range(1, int(len(nums) / 2)):
        step = do_check(x)
        if step != -1:
            step_list.append(step)
    if len(step_list) > 0:
        step_list.sort()
        return step_list[0]
    else:
        return -1


def fun_5(num):
    """
    用连续自然数之和来表达整数
    一个整数可以由连续的自然数之和来表示, 给定一个整数, 计算该整数有几种连续自然数之和的
    9=4+5
    9=2+3+4

    10=1+2+3+4
    """
    res = []
    for n in range(1, num):
        total = 0
        temp = ""
        for i in range(n, num):
            if total > num:
                break
            total+=i
            temp += str(i) + "+"
            if total == num:
                res.append(str(num) + "=" + temp[:-1])
                break
    res.sort(key=lambda y:len(y))
    for x in res:
        print(x)

def fun_6(m):
    """
    https://blog.csdn.net/weixin_44219664/article/details/124029080
    找城市 给定一个 n 个节点
    给定一个 n 个节点的邻接矩阵 m。 节点定义为城市，如果 a 城市与 b 城市相连，
    b 与 c 城市相连，尽管 a 与 c 并不直接相连，但可以认为 a 与 c 相连，
    定义 a,b,c 是一个城市群。
    矩阵 m[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，否则表示不相连。
    请你找出共有多少个城市群。
    输入：[[1,1,0],[1,1,0],[0,0,1]]
    输出：2
    说明： 1 2 相连，3 独立，因此是 2 个城市群。
    """
    def dfs(node):
        visited[node] = 1
        arr = m[node]
        for j in range(len(arr)):
            if arr[j] !=0 and visited[j] == 0:
                dfs(j)

    n = len(m)
    visited = [0]*n
    ans = 0
    for i in range(n):
        if visited[i] == 0:
            ans +=1
            dfs(i)
    return ans

def fun_7(num):
    """
    整数编码 实现一个整数编码方法
    实现一个整数编码方法
    使得待编码的数字越小
    编码后所占用的字节数越小
    编码规则如下
    1.编码时7位一组，每个字节的低7位用于存储待编码数字的补码
    2.字节的最高位表示后续是否还有字节，置1表示后面还有更多的字节， 置0表示当前字节为最后一个字节
    3.采用小端序编码，低位和低字节放在低地址上
    4.编码结果按16进制数的字符格式进行输出，小写字母需要转化为大写字母
    """
    binary = bin(num)[2:]
    binary_size = len(binary)
    print(binary)
    index = binary_size
    res = ""
    while index > 0:
        start = max(index - 7, 0)
        bin_str = binary[start:index]
        if len(bin_str) < 7:
            head = ""
            for j in range(0, 7-len(bin_str)):
                head+="0"
            bin_str = head+bin_str
        bin_str = "0" + bin_str if index - 7 <= 0 else "1" + bin_str
        hex_str = hex(int(bin_str, 2))[2:].upper()
        if len(hex_str) == 1:
            hex_str = "0" + hex_str
        res+=hex_str
        index -= 7

    return res


if __name__ == "__main__":
    aa = fun_7(1000)
    print(aa)

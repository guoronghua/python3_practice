def fun_01():
    """
    字符串最后一个单词的长度
    计算字符串最后一个单词的长度，单词以空格隔开，字符串长度小于5000。（注：字符串末尾不以空格为结尾）
    https://www.nowcoder.com/practice/8c949ea5f36f422594b306a2300315da?tpId=37&tqId=21224&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入：
    hello nowcoder
    输出：
    8
    说明：
    最后一个单词为nowcoder，长度为8
    """
    aa = input().strip().split()
    print(len(aa[-1]))


def fun_02():
    """
    计算某字符出现次数
    https://www.nowcoder.com/practice/a35ce98431874e3a820dbe4b2d0508b1?tpId=37&tqId=21225&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    """
    x = input().lower()
    y = input().lower()
    print(x.count(y))


def fun_03():
    """
    明明的随机数
    https://www.nowcoder.com/practice/3245215fffb84b7b81285493eae92ff0?tpId=37&tqId=21226&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    明明想在学校中请一些同学一起做一项问卷调查，为了实验的客观性，他先用计算机生成了 N 个 1 到 1000 之间的随机整数（ N≤1000 ），对于其中重复的数字，只保留一个，把其余相同的数去掉，不同的数对应着不同的学生的学号。然后再把这些数从小到大排序，按照排好的顺序去找同学做调查。现在明明把他已经用计算机生成好的 N 个随机数按照下面的输入描述的格式交给你，请你协助明明完成“去重”与“排序”的工作。
    注：测试用例保证输入参数的正确性，答题者无需验证。

    数据范围：1≤n≤1000  ，输入的数字大小满足 1≤val≤500
    输入：
    3
    2
    2
    1
    输出：
    1
    2
    说明：
    输入解释：
    第一个数字是3，也即这个小样例的N=3，说明用计算机生成了3个1到1000之间的随机整数，接下来每行一个随机数字，共3行，也即这3个随机数字为：
    2
    2
    1
    所以样例的输出为：
    1
    2
    """
    while True:
        try:
            n = input()  # 指定为N个数，输入
            lst = []  # 指定一个空列表
            for i in range(int(n)):  # 循环N次
                lst.append(int(input()))  # 空集合中追加一个N个数中的某一个随机数
            uni = set(lst)  # 列表去重，但是会变成无序
            lst = list(uni)  # 集合转列表
            lst.sort()  # 列表排序
            for i in lst:
                print(i)  # 打印列表
        except:
            break


def fun_04():
    """
    字符串分隔
    https://www.nowcoder.com/practice/d9162298cb5a437aad722fccccaae8a7?tpId=37&tqId=21227&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    连续输入字符串，请按长度为8拆分每个输入字符串并进行输出；
    •长度不是8整数倍的字符串请在后面补数字0，空字符串不处理。
    输入：
    abc
    输出：
    abc00000
    """
    while True:
        try:
            l = input()
            for i in range(0, len(l), 8):
                print("{:0<8s}".format(l[
                                       i:i + 8]))  # format 格式。表示 输出0~8位字符，“<” 表示左对齐，冒号前面的0是格式化编号，有多个输出的时候标为{0}，{1}；冒号后的0表示以0填充其余位置，s是字符串格式输出的意思
        except:
            break


def fun_05():
    """
    进制转换
    https://www.nowcoder.com/practice/8f3df50d2b9043208c5eed283d1d4da6?tpId=37&tqId=21228&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    写出一个程序，接受一个十六进制的数，输出该数值的十进制表示。
    输入：
    0xAA

    输出：
    170
    """
    temp = [str(x) for x in range(10)]
    temp.extend(["A", "B", "C", "D", "E", "F"])
    while True:
        try:
            a = input()
            res = 0
            for x in range(len(a)):
                res += 16 ** (len(a) - x - 1) * int(temp.index(a[x]))
            print(res)
        except:
            break


def fun_06():
    """
    https://www.nowcoder.com/practice/196534628ca6490ebce2e336b47b3607?tpId=37&tqId=21229&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    质数因子
    输入一个正整数，按照从小到大的顺序输出它的所有质因子（重复的也要列举）（如180的质因子为2 2 3 3 5 ）
    输入：
    180

    输出：
    2 2 3 3 5
    """
    n = int(input())
    for i in range(2, int(n ** 0.5 + 1)):
        while n % i == 0:
            print(i, end=' ')
            n = n // i
    if n > 2:
        print(n)


def fun_07():
    """
    https://www.nowcoder.com/practice/3ab09737afb645cc82c35d56a5ce802a?tpId=37&tqId=21230&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    取近似值
    写出一个程序，接受一个正浮点数值，输出该数值的近似整数值。如果小数点后数值大于等于 0.5 ,向上取整；小于 0.5 ，则向下取整。
    数据范围：保证输入的数字在 32 位浮点数范围内
    :return:
    """
    n = float(input())
    y = lambda x: int(x + 0.5)
    print(y(n))


def fun_08():
    """
    合并表记录
    https://www.nowcoder.com/practice/de044e89123f4a7482bd2b214a685201?tpId=37&tqId=21231&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    数据表记录包含表索引index和数值value（int范围的正整数），请对表索引相同的记录进行合并，即将相同索引的数值进行求和运算，输出按照index值升序进行输出。

    :return:
    """
    n = int(input())
    dic = {}

    # idea: 动态建构字典
    for i in range(n):
        line = input().split()
        key = int(line[0])
        value = int(line[1])
        dic[key] = dic.get(key, 0) + value  # 累积key所对应的value

    for each in sorted(dic):  # 最后的键值对按照升值排序
        print(each, dic[each])


def fun_09():
    """
    提取不重复的整数
    https://www.nowcoder.com/practice/253986e66d114d378ae8de2e6c4577c1?tpId=37&tqId=21232&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking

    输入一个 int 型整数，按照从右向左的阅读顺序，返回一个不含重复数字的新的整数。
    保证输入的整数最后一位不是 0 。
    """
    a = input()[::-1]
    res = ""
    for i in a:
        if i not in res:
            res += i
    print(res)


def fun_10():
    """
    字符个数统计
    https://www.nowcoder.com/practice/eb94f6a5b2ba49c6ac72d40b5ce95f50?tpId=37&tqId=21233&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    编写一个函数，计算字符串中含有的不同字符的个数。字符在 ASCII 码范围内( 0~127 ，包括 0 和 127 )，换行表示结束符，不算在字符里。不在范围内的不作统计。多个相同的字符只计算一次
    例如，对于字符串 abaca 而言，有 a、b、c 三种不同的字符，因此输出 3 。
    """
    str = input()
    string = ''.join(set(str))  # 去重后以字符串的形式
    count = 0  # 开始计数
    for item in string:
        if 0 <= ord(item) <= 127:  # ASCII码范围要求
            count += 1  # 计数
    print(count)


def fun_11():
    """
    数字颠倒
    https://www.nowcoder.com/practice/ae809795fca34687a48b172186e3dafe?tpId=37&tqId=21234&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入一个整数，将这个整数以字符串的形式逆序输出
    程序不考虑负数的情况，若数字含有0，则逆序形式也含有0，如输入为100，则输出为001
    """
    a = input()
    print(a[::-1])


def fun_12():
    """
    字符串排序
    https://www.nowcoder.com/practice/5af18ba2eb45443aa91a11e848aa6723?tpId=37&tqId=21237&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    给定 n 个字符串，请对 n 个字符串按照字典序排列。
    """
    while True:
        try:
            num = int(input())
            stack = []
            for i in range(num):
                stack.append(input())
            print("\n".join(sorted(stack)))
        except:
            break


def fun_13():
    """
    求int型正整数在内存中存储时1的个数
    输入一个 int 型的正整数，计算出该 int 型数据在内存中存储时 1 的个数。
    https://www.nowcoder.com/practice/440f16e490a0404786865e99c6ad91c9?tpId=37&tqId=21238&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    """
    a = int(input())
    res = 0
    while a != 0:
        a, reminder = divmod(a, 2)
        if reminder == 1:
            res += 1
    print(res)

    print(bin(a).count('1'))


def fun_14():
    """
    购物单
    https://www.nowcoder.com/practice/f9c6f980eeec43ef85be20755ddbeaf4?tpId=37&tqId=21239&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    王强决定把年终奖用于购物，他把想买的物品分为两类：主件与附件，附件是从属于某个主件的，下表就是一些主件与附件的例子：

    主件	  附件
    电脑	  打印机，扫描仪
    书柜	  图书
    书桌	  台灯，文具
    工作椅 无

    如果要买归类为附件的物品，必须先买该附件所属的主件，且每件物品只能购买一次。
    每个主件可以有 0 个、 1 个或 2 个附件。附件不再有从属于自己的附件。
    王强查到了每件物品的价格（都是 10 元的整数倍），而他只有 N 元的预算。除此之外，他给每件物品规定了一个重要度，用整数 1 ~ 5 表示。他希望在花费不超过 N 元的前提下，使自己的满意度达到最大。
    满意度是指所购买的每件物品的价格与重要度的乘积的总和，假设设第ii件物品的价格为v[i]v[i]，重要度为w[i]w[i]，共选中了kk件物品，编号依次为j_1,j_2,...,j_kj     ​

    输入描述：
    输入的第 1 行，为两个正整数N，m，用一个空格隔开：
    （其中 N （ N<32000 ）表示总钱数， m （m <60 ）为可购买的物品的个数。）
    从第 2 行到第 m+1 行，第 j 行给出了编号为 j-1 的物品的基本数据，每行有 3 个非负整数 v p q
    （其中 v 表示该物品的价格（ v<10000 ）， p 表示该物品的重要度（ 1 ~ 5 ）， q 表示该物品是主件还是附件。如果 q=0 ，表示该物品为主件，如果 q>0 ，表示该物品为附件， q 是所属主件的编号）
    """
    from collections import defaultdict

    # 处理输入
    n, m = map(int, input().split())
    n //= 10  # 价格总为 10 的倍数，优化空间复杂度
    prices = defaultdict(lambda: [0, 0, 0])  # 主从物品的价格
    values = defaultdict(lambda: [0, 0, 0])  # 主从物品的价值

    for i in range(1, m + 1):  # i 代表第 i 个物品
        v, p, q = map(int, input().split())
        v //= 10  # 价格总为 10 的倍数，优化空间复杂度
        if q == 0:  # 追加主物品
            prices[i][0] = v
            values[i][0] = v * p
        elif prices[q][1]:  # 追加从物品
            prices[q][2] = v
            values[q][2] = v * p
        else:
            prices[q][1] = v
            values[q][1] = v * p

    # 处理输出
    dp = [0] * (n + 1)  # 初始化 dp 数组

    for i, v in prices.items():
        for j in range(n, v[0] - 1, -1):
            p1, p2, p3 = v
            v1, v2, v3 = values[i]
            # 处理主从组合的四种情况
            dp[j] = max(dp[j], dp[j - p1] + v1)  # 主件
            dp[j] = max(dp[j], dp[j - p1 - p2] + v1 + v2) if j >= p1 + p2 else dp[j]  # 主件 + 附件1
            dp[j] = max(dp[j], dp[j - p1 - p3] + v1 + v3) if j >= p1 + p3 else dp[j]  # 主件 + 附件2
            dp[j] = max(dp[j], dp[j - p1 - p2 - p3] + v1 + v2 + v3) if j >= p1 + p2 + p3 else dp[j]  # 主件 + 附件1 + 附件2
    print(dp[n] * 10)


def fun_15():
    """
    坐标移动
    https://www.nowcoder.com/practice/119bcca3befb405fbe58abe9c532eb29?tpId=37&tqId=21240&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    开发一个坐标计算工具， A表示向左移动，D表示向右移动，W表示向上移动，S表示向下移动。从（0,0）点开始移动，从输入字符串里面读取一些坐标，并将最终输入结果输出到输出文件里面。
    合法坐标为A(或者D或者W或者S) + 数字（两位以内）
    坐标之间以;分隔。
    非法坐标点需要进行丢弃。如AA10;  A1A;  $%$;  YAD; 等。
    """
    a = input()
    arr = list(filter(lambda x: str(x[1:]).isdigit(), a.split(";")))
    point = [0, 0]
    for x in arr:
        if x[0].upper() == "A":
            point[0] -= int(x[1:])
        if x[0].upper() == "D":
            point[0] += int(x[1:])
        if x[0].upper() == "S":
            point[1] -= int(x[1:])
        if x[0].upper() == "W":
            point[1] += int(x[1:])
    print("%s,%s" % (point[0], point[1]))


def fun_16():
    """
    识别有效的IP地址和掩码并进行分类统计
    https://www.nowcoder.com/practice/de538edd6f7e4bc3a5689723a7435682?tpId=37&tqId=21241&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    请解析IP地址和对应的掩码，进行分类识别。要求按照A/B/C/D/E类地址归类，不合法的地址和掩码单独归类。

    所有的IP地址划分为 A,B,C,D,E五类
    A类地址从1.0.0.0到126.255.255.255;
    B类地址从128.0.0.0到191.255.255.255;
    C类地址从192.0.0.0到223.255.255.255;
    D类地址从224.0.0.0到239.255.255.255；
    E类地址从240.0.0.0到255.255.255.255

    私网IP范围是：
    从10.0.0.0到10.255.255.255
    从172.16.0.0到172.31.255.255
    从192.168.0.0到192.168.255.255

    子网掩码为二进制下前面是连续的1，然后全是0。（例如：255.255.255.32就是一个非法的掩码）
    （注意二进制下全是1或者全是0均为非法子网掩码）

    注意：
    1. 类似于【0.*.*.*】和【127.*.*.*】的IP地址不属于上述输入的任意一类，也不属于不合法ip地址，计数时请忽略
    2. 私有IP地址和A,B,C,D,E类地址是不冲突的

    输入描述：
    多行字符串。每行一个IP地址和掩码，用~隔开。

    输出描述：
    统计A、B、C、D、E、错误IP地址或错误掩码、私有IP的个数，之间以空格隔开。
    """
    ipClass2num = {
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0,
        'E': 0,
        'ERROR': 0,
        'PRIVATE': 0,
    }

    # 私有IP地址和A,B,C,D,E类地址是不冲突的，也就是说需要同时+1
    def check_ip(ip: str):
        ip_bit = ip.split('.')
        if len(ip_bit) != 4 or '' in ip_bit:  # ip 的长度为4 且每一位不为空
            return False
        for i in ip_bit:
            if int(i) < 0 or int(i) > 255:  # 检查Ip每一个10位的值范围为0~255
                return False
        return True

    def check_mask(mask: str):
        if not check_ip(mask):
            return False
        if mask == '255.255.255.255' or mask == '0.0.0.0':
            return False
        mask_list = mask.split('.')
        if len(mask_list) != 4:
            return False
        mask_bit = []
        for i in mask_list:  # 小数点隔开的每一数字段
            i = bin(int(i))  # 每一数字段转换为每一段的二进制数字段
            i = i[2:]  # 从每一段的二进制数字段的第三个数开始，因为前面有两个ob
            mask_bit.append(i.zfill(8))  # .zfill:返回指定长度的字符串，原字符串右对齐，前面填充0
        whole_mask = ''.join(mask_bit)
        #     print(whole_mask)
        whole0_find = whole_mask.find("0")  # 查0从哪里开始
        whole1_rfind = whole_mask.rfind("1")  # 查1在哪里结束
        if whole1_rfind + 1 == whole0_find:  # 两者位置差1位为正确
            return True
        else:
            return False

    def check_private_ip(ip: str):
        # 三类私网
        ip_list = ip.split('.')
        if ip_list[0] == '10': return True
        if ip_list[0] == '172' and 16 <= int(ip_list[1]) <= 31: return True
        if ip_list[0] == '192' and ip_list[1] == '168': return True
        return False

    while True:
        try:
            s = input()
            ip = s.split('~')[0]
            mask = s.split('~')[1]
            if check_ip(ip):
                first = int(ip.split('.')[0])
                if first == 127 or first == 0:
                    # 若不这样写，当类似于【0.*.*.*】和【127.*.*.*】的IP地址的子网掩码错误时，会额外计数
                    continue
                if check_mask(mask):
                    if check_private_ip(ip):
                        ipClass2num['PRIVATE'] += 1
                    if 0 < first < 127:
                        ipClass2num['A'] += 1
                    elif 127 < first <= 191:
                        ipClass2num['B'] += 1
                    elif 192 <= first <= 223:
                        ipClass2num['C'] += 1
                    elif 224 <= first <= 239:
                        ipClass2num['D'] += 1
                    elif 240 <= first <= 255:
                        ipClass2num['E'] += 1
                else:
                    ipClass2num['ERROR'] += 1
            else:
                ipClass2num['ERROR'] += 1
        except:
            break
    for v in ipClass2num.values():
        print(v, end=(' '))


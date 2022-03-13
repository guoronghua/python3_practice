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


def fun_17():
    """
    密码验证合格程序
    https://www.nowcoder.com/practice/184edec193864f0985ad2684fbc86841?tpId=37&tqId=21243&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    密码要求:

    1.长度超过8位
    2.包括大小写字母.数字.其它符号,以上四种至少三种
    3.不能有长度大于2的不含公共元素的子串重复 （注：其他符号不含空格或换行）
    """

    def check(s):
        if len(s) <= 8:
            return 0
        a, b, c, d = 0, 0, 0, 0
        for item in s:
            if ord('a') <= ord(item) <= ord('z'):
                a = 1
            elif ord('A') <= ord(item) <= ord('Z'):
                b = 1
            elif ord('0') <= ord(item) <= ord('9'):
                c = 1
            else:
                d = 1
        if a + b + c + d < 3:
            return 0
        # or repeat_sub = re.findall(r'(.{3,}).*\1', line) \1表示与()里的内容重复
        for i in range(len(s) - 3):
            if len(s.split(s[i:i + 3])) >= 3:
                return 0
        return 1

    while 1:
        try:
            print('OK' if check(input()) else 'NG')
        except:
            break


def fun_18():
    """
    简单密码
    https://www.nowcoder.com/practice/7960b5038a2142a18e27e4c733855dac?tpId=37&tqId=21244&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking

    现在有一种密码变换算法。
    九键手机键盘上的数字与字母的对应： 1--1， abc--2, def--3, ghi--4, jkl--5, mno--6, pqrs--7, tuv--8 wxyz--9, 0--0，把密码中出现的小写字母都变成九键键盘对应的数字，如：a 变成 2，x 变成 9.
    而密码中出现的大写字母则变成小写之后往后移一位，如：X ，先变成小写，再往后移一位，变成了 y ，例外：Z 往后移是 a 。
    数字和其它的符号都不做变换。
    """
    a = input()
    res = ""
    d = {
        "abc": 2,
        "def": 3,
        "ghi": 4,
        "jkl": 5,
        "mno": 6,
        "pqrs": 7,
        "tuv": 8,
        "wxyz": 9,

    }
    for x in a:
        if str(x).isupper():
            if str(x).lower() == "z":
                res += "a"
            else:
                res += chr(int(ord(str(x).lower())) + 1)
        elif str(x).islower():
            for y in d.keys():
                if x in y:
                    res += str(d.get(y))
        else:
            res += str(x)
    print(res)


def fun_19():
    """
    汽水瓶
    https://www.nowcoder.com/practice/fe298c55694f4ed39e256170ff2c205f?tpId=37&tqId=21245&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking

    某商店规定：三个空汽水瓶可以换一瓶汽水，允许向老板借空汽水瓶（但是必须要归还）。
    小张手上有n个空汽水瓶，她想知道自己最多可以喝到多少瓶汽水。
    数据范围：输入的正整数满足 1 \le n \le 100 \1≤n≤100
    """
    import sys
    lines = [line.rstrip("\n") for line in sys.stdin.readlines()]
    for line in lines:
        k = int(line)
        res = 0
        while k > 2:
            beer, reminder_k = divmod(k, 3)
            res += beer
            k = beer + reminder_k
        if k == 2:
            res += 1
        print(res)


def fun_20():
    """
    删除字符串中出现次数最少的字符
    https://www.nowcoder.com/practice/05182d328eb848dda7fdd5e029a56da9?tpId=37&tqId=21246&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    实现删除字符串中出现次数最少的字符，若出现次数最少的字符有多个，则把出现次数最少的字符都删除。输出删除这些单词后的字符串，字符串中其它字符保持原来的顺序。
    """
    import sys
    from collections import Counter

    lines = [line.rstrip("\n") for line in sys.stdin.readlines()]
    for line in lines:
        c = Counter(line)
        min_length = min(c.values())
        for k, v in c.items():
            if v == min_length:
                line = line.replace(k, "")
        print(line)


def fun_21():
    """
    合唱队
    https://www.nowcoder.com/practice/6d9d69e3898f45169a441632b325c7b4?tpId=37&tqId=21247&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    N 位同学站成一排，音乐老师要请最少的同学出列，使得剩下的 K 位同学排成合唱队形。
    通俗来说，能找到一个同学，他的两边的同学身高都依次严格降低的队形就是合唱队形。
    例子：
    123 124 125 123 121 是一个合唱队形
    123 123 124 122不是合唱队形，因为前两名同学身高相等，不符合要求
    123 122 121 122不是合唱队形，因为找不到一个同学，他的两侧同学身高递减。
    你的任务是，已知所有N位同学的身高，计算最少需要几位同学出列，可以使得剩下的同学排成合唱队形。
    注意：不允许改变队列元素的先后顺序 且 不要求最高同学左右人数必须相等

    """
    import bisect  # 引入二分法
    def hcteam(l):  # 定义一个函数，寻找最长的子序列
        arr = [l[0]]  # 定义列表，将传入函数的列表第一个元素放入当前元素
        dp = [1] * len(l)  # 定义一个列表，默认子序列有当前元素1，长度是传入函数的列表长度
        for i in range(1, len(l)):  # 从第二个元素开始查找
            if l[i] > arr[-1]:  # 如果元素大于arr列表的最后一个元素，就把它插入列表末尾
                arr.append(l[i])
                dp[i] = len(arr)  # 获取这个元素子序列的长度
            else:  # 否则，利用二分法找到比元素大的元素的位置，用新的元素替代比它大的那个元素的值，这样就能制造出一个顺序排列的子序列
                pos = bisect.bisect_left(arr, l[i])
                arr[pos] = l[i]
                dp[i] = pos + 1  # 获取这个元素子序列的长度
        return dp

    while True:
        try:
            n = int(input())
            sg = list(map(int, input().split()))
            left_t = hcteam(sg)  # 向左遍历查找子序列
            right_t = hcteam(sg[::-1])[::-1]  # 向右遍历查找子序列
            res = [left_t[i] + right_t[i] - 1 for i in range(len(sg))]  # 因为左右都包含原元素，所以需要减1 ；得到各元素能得到的子序列的最大长度
            print(n - max(res))  # 源列表长度-可以生成的最长子序列长度  得到需要剔除的最小人数
        except:
            break

    # 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
    def length_of_lst(nums):
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


def fun_22():
    """
    字符串排序
    https://www.nowcoder.com/practice/5190a1db6f4f4ddb92fd9c365c944584?tpId=37&tqId=21249&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    编写一个程序，将输入字符串中的字符按如下规则排序。

    规则 1 ：英文字母从 A 到 Z 排列，不区分大小写。
    如，输入： Type 输出： epTy

    规则 2 ：同一个英文字母的大小写同时存在时，按照输入顺序排列。
    如，输入： BabA 输出： aABb

    规则 3 ：非英文字母的其它字符保持原来的位置。
    如，输入： By?e 输出： Be?y
    """
    while True:
        try:
            s = input()
            a = ''
            for i in s:
                if i.isalpha():
                    a += i
            b = sorted(a, key=str.upper)
            index = 0
            d = ''
            for i in range(len(s)):
                if s[i].isalpha():
                    d += b[index]
                    index += 1
                else:
                    d += s[i]
            print(d)
        except:
            break


def fun_23():
    """
    素数伴侣
    https://www.nowcoder.com/practice/b9eae162e02f4f928eac37d7699b352e?tpId=37&tqId=21251&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking

    题目描述
    若两个正整数的和为素数，则这两个正整数称之为“素数伴侣”，如2和5、6和13，它们能应用于通信加密。现在密码学会请你设计一个程序，从已有的 N （ N 为偶数）个正整数中挑选出若干对组成“素数伴侣”，挑选方案多种多样，例如有4个正整数：2，5，6，13，如果将5和6分为一组中只能得到一组“素数伴侣”，而将2和5、6和13编组将得到两组“素数伴侣”，能组成“素数伴侣”最多的方案称为“最佳方案”，当然密码学会希望你寻找出“最佳方案”。

    输入:
    有一个正偶数 n ，表示待挑选的自然数的个数。后面给出 n 个具体的数字。

    输出:
    输出一个整数 K ，表示你求得的“最佳方案”组成“素数伴侣”的对数。

    解题思路： 1.判断一个数是否为素数：在这里需要注意的是全部数判断会超时，所以选择半位数来判断可以节省时间。选用原理：一个数能被另一个数除尽，则这个数也能被它的2倍数除尽，一个数不能被另一个数除尽则也不能被它的2倍数除尽。 2.判断最大匹配数： 这个是参考已有的答案来的，一个列表里的数与另一个列表里的每一个数判断，如果他们的和是素数就标记，若这个数还有另一个匹配的数，则看看之前匹配的数是否还有其他匹配的数，有则这个数就被当前数替代，没有则跳过。如此一来得到的数即为最大匹配数 3.数的分配与具体判断实现： 奇数和奇数相加与偶数和偶数相加得到的是偶数不可能是素数，只能是奇数和偶数相加才可能存在素数。因此将所有可匹配的数按奇数和偶数分为两个列表。然后让每一个奇数与所有偶数列表的数去匹配看相加的和是否为素数，如果是则加1。最终将计算的数打印出来

    """

    def get_primenum(s):
        if s < 4:
            return True
        # 通过从2到它的平方根之间没有可除尽的数来判断这个数是否为素数。原理：一个数与另一个数能除尽则也能除尽这个数的2倍数。若直接判断从2到这个数之间的数则会耗费大量的时间来计算导致超时。
        for i in range(2, int(s ** 0.5) + 1):
            if s % i == 0:
                return False
        return True

    def find_even(evens, previous_select, final_select, odd):
        for i, even in enumerate(evens):
            if get_primenum(even + odd) and previous_select[i] == 0:
                previous_select[i] = 1
                # 判断第i位偶数是否被匹配或者它的匹配奇数是否有其他选择，如果有其他选择，则当前的奇数匹配第i位偶数
                if final_select[i] == 0 or find_even(evens, previous_select, final_select, final_select[i]):
                    final_select[i] = odd
                    return True
        return False

    while True:
        try:
            list0 = list(map(int, input().split(' ')))
            count0 = 0
            evens, odds = [], []
            for list1 in list0:
                if list1 % 2 == 0:
                    evens.append(list1)
                else:
                    odds.append(list1)
            final_select = [0] * len(evens)
            for odd in odds:
                previous_select = [0] * len(evens)
                if find_even(evens, previous_select, final_select, odd):
                    count0 += 1
            print(count0)
        except:
            break

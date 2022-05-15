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
                print("{0:0<8s}".format(l[i:i + 8]))  # format 格式。表示 输出0~8位字符，“<” 表示左对齐，冒号前面的0是格式化编号，有多个输出的时候标为{0}，{1}；冒号后的0表示以0填充其余位置，s是字符串格式输出的意思
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
    满意度是指所购买的每件物品的价格与重要度的乘积的总和，假设设第ii件物品的价格为v[i]，重要度为w[i]，共选中了kk件物品，编号依次为j_1,j_2,...,j_kj     ​

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
    prices = defaultdict(lambda: [0, 0, 0])  # 0，0，0 主从从 物品的价格
    values = defaultdict(lambda: [0, 0, 0])  # 0，0，0 主从从 物品的价值

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

    for i, v in prices.items():  # 遍历每一个物品价格
        p1, p2, p3 = v          # 价值
        v1, v2, v3 = values[i]  # 满意度
        for j in range(n, p1 - 1, -1):  # 价格从n到当前主物的价格v[0]遍历
            # 处理主从组合的四种情况
            dp[j] = max(dp[j], dp[j - p1] + v1)  # 只选主件的满意度
            dp[j] = max(dp[j], dp[j - p1 - p2] + v1 + v2) if j >= p1 + p2 else dp[j]  # 主件 + 附件1的满意度
            dp[j] = max(dp[j], dp[j - p1 - p3] + v1 + v3) if j >= p1 + p3 else dp[j]  # 主件 + 附件2的满意度
            dp[j] = max(dp[j], dp[j - p1 - p2 - p3] + v1 + v2 + v3) if j >= p1 + p2 + p3 else dp[j]  # 主件 + 附件1 + 附件2的满意度
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
    123 123 124 122 不是合唱队形，因为前两名同学身高相等，不符合要求
    123 122 121 122 不是合唱队形，因为找不到一个同学，他的两侧同学身高递减。
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

    解题思路： 1.判断一个数是否为素数：在这里需要注意的是全部数判断会超时，所以选择半位数来判断可以节省时间。选用原理：一个数能被另一个数除尽，则这个数也能被它的2倍数除尽，
    一个数不能被另一个数除尽则也不能被它的2倍数除尽。 2.判断最大匹配数： 这个是参考已有的答案来的，一个列表里的数与另一个列表里的每一个数判断，如果他们的和是素数就标记，
    若这个数还有另一个匹配的数，则看看之前匹配的数是否还有其他匹配的数，有则这个数就被当前数替代，没有则跳过。如此一来得到的数即为最大匹配数 3.数的分配与具体判断实现：
    奇数和奇数相加与偶数和偶数相加得到的是偶数不可能是素数，只能是奇数和偶数相加才可能存在素数。因此将所有可匹配的数按奇数和偶数分为两个列表。然后让每一个奇数与所有偶数列表的数去匹配看相加的和是否为素数，
    如果是则加1。最终将计算的数打印出来

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


def fun_24():
    """
    密码截取，查找最长回文子串
    https://www.nowcoder.com/practice/3cd4621963e8454594f00199f4536bb1?tpId=37&tqId=21255&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    Catcher是MCA国的情报员，他工作时发现敌国会用一些对称的密码进行通信，比如像这些ABBA，ABA，A，123321，但是他们有时会在开始或结束时加入一些无关的字符以防止别国破解。
    比如进行下列变化 ABBA->12ABBA,ABA->ABAKK,123321->51233214　。因为截获的串太长了，而且存在多种可能的情况（abaaab可看作是aba,或baaab的加密形式），Cathcer的工作量实在是太大了，
    他只能向电脑高手求助，你能帮Catcher找出最长的有效密码串吗？
    """

    def longp(s):
        res = ''
        for i in range(len(s)):
            # 先判定奇数的，从i开始左右对比
            tmp = helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # 再判定偶数的，从i和i+1开始对比
            tmp = helper(s, i, i + 1)
            if len(tmp) > len(res):
                res = tmp
        print(len(res))

    def helper(s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]

    while True:
        try:
            s = input()
            longp(s)
        except:
            break


def fun_25():
    """
    整数与IP地址间的转换
    https://www.nowcoder.com/practice/66ca0e28f90c42a196afd78cc9c496ea?tpId=37&tqId=21256&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入
    1 输入IP地址
    2 输入10进制型的IP地址

    输出描述：
    输出
    1 输出转换成10进制的IP地址
    2 输出转换后的IP地址
    """
    while True:
        try:
            ip = input()
            num = input()
            ip_arr = ip.split(".")

            num_resp = ""
            ip_res = []

            num_flag = True
            for x in ip_arr:
                if int(x) > 255:
                    num_flag = False
                    continue
                bin_num = bin(int(x)).replace("0b", "").rjust(8, '0')
                num_resp += bin_num

            if num_flag:
                num_resp = str(int(num_resp, 2))
                print(num_resp)

            bin_num = bin(int(num)).replace("0b", "").rjust(32, '0')
            for i in range(4):
                ip_res.append(str(int(bin_num[8 * i:(i + 1) * 8], 2)))
            if len(bin_num) <= 32:
                print(".".join(ip_res))
        except:
            break


def fun_26():
    """
    蛇形矩阵
    https://www.nowcoder.com/practice/649b210ef44446e3b1cd1be6fa4cab5e?tpId=37&tqId=21258&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    蛇形矩阵是由1开始的自然数依次排列成的一个矩阵上三角形。

    例如，当输入5时，应该输出的三角形为：
    1 3 6 10 15
    2 5 9 14
    4 8 13
    7 12
    11
    """
    while True:
        try:
            a = int(input())
            count = 1
            result = [[None] * a for _ in range(a)]

            for x in range(a):
                for y in range(x + 1):
                    temp = count
                    if x == 0:
                        result[x][y] = temp
                    else:
                        result[y][x - y] = temp
                    count += 1

            for y in range(a):
                line_str = ""
                for x in result:
                    if x[y]:
                        line_str = line_str + str(x[y]) + " "
                line_str = line_str.rstrip(" ")
                print(line_str)
        except:
            break


def fun_27():
    """
    统计每个月兔子的总数
    https://www.nowcoder.com/practice/1221ec77125d4370833fd3ad5ba72395?tpId=37&tqId=21260&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    有一种兔子，从出生后第3个月起每个月都生一只兔子，小兔子长到第三个月后每个月又生一只兔子。
    例子：假设一只兔子第3个月出生，那么它第5个月开始会每个月生一只兔子。
    假如兔子都不死，问第n个月的兔子总数为多少？
    """

    def count(a):
        if a <= 2:
            return 1
        return count(a - 1) + count(a - 2)

    while True:
        try:
            a = int(input())
            arr = [1, 1]
            while len(arr) < a:
                arr.append(arr[-1] + arr[-2])
            print(arr[-1])
        except:
            break


def fun_28():
    """
    判断两个IP是否属于同一子网
    https://www.nowcoder.com/practice/34a597ee15eb4fa2b956f4c595f03218?tpId=37&tqId=21262&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    IP地址是由4个0-255之间的整数构成的，用"."符号相连。
    二进制的IP地址格式有32位，例如：10000011，01101011，00000011，00011000;每八位用十进制表示就是131.107.3.24
    子网掩码是用来判断任意两台计算机的IP地址是否属于同一子网络的根据。
    子网掩码与IP地址结构相同，是32位二进制数，由1和0组成，且1和0分别连续，其中网络号部分全为“1”和主机号部分全为“0”。
    你可以简单的认为子网掩码是一串连续的1和一串连续的0拼接而成的32位二进制数，左边部分都是1，右边部分都是0。
    利用子网掩码可以判断两台主机是否中同一子网中。
    若两台主机的IP地址分别与它们的子网掩码进行逻辑“与”运算（按位与/AND）后的结果相同，则说明这两台主机在同一子网中。
    """
    while True:
        try:
            x = input().split('.')
            y = input().split('.')
            z = input().split('.')
            m, n = [], []
            for i in range(len(x)):
                x[i] = int(x[i])
                y[i] = int(y[i])
                z[i] = int(z[i])
            if x[0] != 255 or x[3] != 0 or max(x + y + z) > 255 or min(x + y + z) < 0:
                print('1')
            else:
                for i in range(len(x)):
                    m.append(int(x[i]) & int(y[i]))
                    n.append(int(x[i]) & int(z[i]))
                if m == n:
                    print('0')
                else:
                    print('2')
        except:
            break


def fun_29():
    """
    称砝码
    https://www.nowcoder.com/practice/f9a4c19050fc477e9e27eb75f3bfd49c?tpId=37&tqId=21264&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    现有n种砝码，重量互不相等，分别为 m1,m2,m3…mn ；
    每种砝码对应的数量为 x1,x2,x3...xn 。现在要用这些砝码去称物体的重量(放在同一侧)，问能称出多少种不同的重量。
    注：
    称重重量包括 0

    对于每组测试数据：
    第一行：n --- 砝码的种数(范围[1,10])
    第二行：m1 m2 m3 ... mn --- 每种砝码的重量(范围[1,2000])
    第三行：x1 x2 x3 .... xn --- 每种砝码对应的数量(范围[1,10])
    利用给定的砝码可以称出的不同的重量数
    """
    while True:
        try:
            weight_type = input()
            weights = list(map(int, input().rstrip("\n").split(" ")))
            weigh_num = list(map(int, input().rstrip("\n").split(" ")))
            groups = []
            for x in range(len(weights)):
                group = set()
                for y in range(0, weigh_num[x] + 1):
                    group.add(weights[x] * y)
                groups.append(group)
            res = {0}
            for group in groups:
                for item in res:
                    test1 = set(w + item for w in group)
                    res = res.union(test1)
            print(len(res))
        except:
            break


def fun_30():
    """
    学英语
    https://www.nowcoder.com/practice/1364723563ab43c99f3d38b5abef83bc?tpId=37&tqId=21265&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    Jessi初学英语，为了快速读出一串数字，编写程序将数字转换成英文：

    1.在英语读法中三位数字看成一整体，后面再加一个计数单位。从最右边往左数，三位一单位，例如12,345 等
    2.每三位数后记得带上计数单位 分别是thousand, million, billion.
    3.公式：百万以下千以上的数 X thousand X, 10亿以下百万以上的数：X million X thousand X, 10 亿以上的数：X billion X million X thousand X. 每个X分别代表三位数或两位数或一位数。
    4.在英式英语中百位数和十位数之间要加and，美式英语中则会省略，我们这个题目采用加上and，百分位为零的话，这道题目我们省略and
    下面再看几个数字例句：
    22: twenty two
    100:  one hundred
    145:  one hundred and forty five
    1,234:  one thousand two hundred and thirty four
    8,088:  eight thousand (and) eighty eight (注:这个and可加可不加，这个题目我们选择不加)
    486,669:  four hundred and eighty six thousand six hundred and sixty nine
    1,652,510:  one million six hundred and fifty two thousand five hundred and ten
    """
    num10 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    num20 = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen"]
    num100 = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    while True:
        try:
            a = input()

            def parse(num):
                if not str(num).isdigit() or int(num) < 0:
                    return "error"
                num = int(num)
                res = ""
                billion = int(num / 1000000000)
                if billion != 0:
                    res += trans(billion) + " billion "
                num %= 1000000000

                million = int(num / 1000000)
                if million != 0:
                    res += trans(million) + " million "
                num %= 1000000

                thousand = int(num / 1000)
                if thousand != 0:
                    res += trans(thousand) + " thousand "
                num %= 1000

                if num != 0:
                    res += trans(num)
                return str(res).rstrip(" ")

            def trans(num):
                res = ""
                h = int(num / 100)   # 处理百位数

                if h != 0:
                    res += num10[h] + " hundred"
                num %= 100

                k = int(num / 10)
                if k != 0:     # 处理十位数
                    if h != 0:
                        res += " and "
                    if k == 1:  # 20以内的
                        res += num20[num % 10]
                    else:
                        res += num100[k - 2] + " " # 超过20
                        if num % 10 != 0:
                            res += num10[num % 10]
                elif (num % 10) != 0:  # 处理个位数
                    if h != 0:
                        res += " and "
                    res += num10[num % 10]
                return str(res).rstrip(" ")

            result = parse(a)
            print(result)
        except Exception as e:
            break


def fun_31():
    """
    迷宫问题
    https://www.nowcoder.com/practice/cf24906056f4488c9ddb132f317e03bc?tpId=37&tqId=21266&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    定义一个二维数组 N*M ，如 5 × 5 数组下所示：

    int maze[5][5] = {
    0, 1, 0, 0, 0,
    0, 1, 1, 1, 0,
    0, 0, 0, 0, 0,
    0, 1, 1, 1, 0,
    0, 0, 0, 1, 0,
    };

    它表示一个迷宫，其中的1表示墙壁，0表示可以走的路，只能横着走或竖着走，不能斜着走，要求编程序找出从左上角到右下角的路线。入口点为[0,0],既第一格是可以走的路。
    """

    # 方法1
    def find_way(x, y, path):
        if x == m - 1 and y == n - 1:
            [print('({%s},{%s})' % (l[0], l[1])) for l in path]

        if x + 1 <= m - 1 and (x + 1, y) not in path and maze[x + 1][y] == '0':
            find_way(x + 1, y, path + [(x + 1, y)])

        if y + 1 <= n - 1 and (x, y + 1) not in path and maze[x][y + 1] == '0':
            find_way(x, y + 1, path + [(x, y + 1)])

        if x - 1 >= 0 and (x - 1, y) not in path and maze[x - 1][y] == '0':
            find_way(x - 1, y, path + [(x - 1, y)])

        if y - 1 >= 0 and (x, y - 1) not in path and maze[x][y - 1] == '0':
            find_way(x, y - 1, path + [(x, y - 1)])

    while 1:
        try:
            m, n = map(int, input().split())
            maze = [input().split() for _ in range(m)]
            find_way(0, 0, [(0, 0)])
        except:
            break

    # 方法2
    while True:
        try:
            m, n = map(int, input().split())
            matrix = []
            for i in range(m):
                d = list(map(int, input().split()))
                # print(c)
                matrix.append(d)

            def dfs(path):
                if path[-1] == [m - 1, n - 1]:
                    return path
                r, c = path[-1]
                matrix[r][c] = 2
                directions = [[r + 1, c], [r, c + 1], [r - 1, c], [r, c - 1]]
                for i, j in directions:
                    if 0 <= i < m and 0 <= j < n and matrix[i][j] == 0:
                        final_path = dfs(path + [[i, j]])
                        if final_path[-1] == [m - 1, n - 1]:
                            return final_path
                return path  # 很重要，保证输出

            path = dfs([[0, 0]])
            for i in path:
                print('({},{})'.format(i[0], i[1]))
        except:
            break


def fun_32():
    """
    解数独
    https://www.nowcoder.com/practice/78a1a4ebe8a34c93aac006c44f6bf8a1?tpId=37&tqId=21267&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    数独（Sudoku）是一款大众喜爱的数字逻辑游戏。玩家需要根据9X9盘面上的已知数字，推算出所有剩余空格的数字，并且满足每一行、每一列、每一个3X3粗线宫内的数字均含1-9，并且不重复。
    """

    def dfs(pos: int):
        global valid
        if pos == len(spaces):
            valid = True
            return

        i, j = spaces[pos]
        for digit in range(9):
            if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
                board[i][j] = str(digit + 1)
                dfs(pos + 1)
                line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = False
            if valid:
                return

    while True:
        try:
            board = []
            for i in range(9):
                row = list(map(int, input().split()))
                board.append(row)

            line = [[False] * 9 for _ in range(9)]
            column = [[False] * 9 for _ in range(9)]
            block = [[[False] * 9 for _ in range(3)] for _ in range(3)]
            valid = False
            spaces = list()
            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        spaces.append((i, j))
                    else:
                        digit = int(board[i][j]) - 1
                        line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
            dfs(0)
            for i in range(9):
                board[i] = list(map(str, board[i]))
                print(' '.join(board[i]))
        except:
            break


def fun_33():
    """
    从单向链表中删除指定值的节点
    https://www.nowcoder.com/practice/f96cd47e812842269058d483a11ced4f?tpId=37&tqId=21271&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入一个单向链表和一个节点的值，从单向链表中删除等于该值的节点，删除后如果链表中无节点则返回空指针。

    链表的值不能重复。
    构造过程，例如输入一行数据为:
    6 2 1 2 3 2 5 1 4 5 7 2 2
    则第一个参数6表示输入总共6个节点，第二个参数2表示头节点值为2，剩下的2个一组表示第2个节点值后面插入第1个节点值，为以下表示:
    1 2 表示为
    2->1
    链表为2->1
    3 2表示为
    2->3
    链表为2->3->1
    5 1表示为
    1->5
    链表为2->3->1->5
    4 5表示为
    5->4
    链表为2->3->1->5->4
    7 2表示为
    2->7
    链表为2->7->3->1->5->4
    最后的链表的顺序为 2 7 3 1 5 4
    """

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

        def travel(self):
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
            while cur:
                if cur.data == target_data:
                    if cur == self._head:
                        self._head = cur.next
                    else:
                        pre.next = cur.next
                    break
                else:
                    pre = cur
                    cur = cur.next

    while True:
        try:
            a = list(map(int, input().split()))
            head_node = Node(a[1])
            nodes = a[2:len(a) - 1]
            to_del_data = a[-1]
            single_link_list = SingleLinkList(head_node)
            for i in range(0, len(nodes), 2):
                data = nodes[i]
                target_data = nodes[i + 1]
                single_link_list.insert(data, target_data)

            single_link_list.remove(to_del_data)
            single_link_list.travel()
        except:
            break


def fun_34():
    """
    四则运算
    https://www.nowcoder.com/practice/9999764a61484d819056f807d2a91f1e?tpId=37&tqId=21273&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入一个表达式（用字符串表示），求这个表达式的值。
    保证字符串中的有效字符包括[‘0’-‘9’],‘+’,‘-’, ‘*’,‘/’ ,‘(’， ‘)’,‘[’, ‘]’,‘{’ ,‘}’。且表达式一定合法。
    输入：
    3+2*{1+2*[-4/(8-6)+7]}
    输出：
    25
    """
    import re
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

    def final_cal(formula_list):
        num_stack = []
        op_stack = []
        for e in formula_list:
            operator = is_operator(e)
            if not operator:
                a = 2
                num_stack.append(float(e))
            else:
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
        print(int(num_stack.pop()))

    while True:
        try:
            a = formula_format(input())
            final_cal(a)
        except:
            break


def fun_35():
    """
    输出单向链表中倒数第k个结点
    https://www.nowcoder.com/practice/54404a78aec1435a81150f15f899417d?tpId=37&tqId=21274&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入一个单向链表，输出该链表中倒数第k个结点，链表的倒数第1个结点为链表的尾指针。
    """

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
            if index > self.count or index == 0:
                print(0)
            else:
                i = 1
                cur = self._head
                while i < index:
                    cur = cur.next
                    i += 1
                print(cur.data)

    while True:
        try:
            node_num = int(input())
            node_valus = [int(x) for x in input().split()]
            target_index = int(input())
            linked_list = SingleLinkedList()
            for x in node_valus:
                node = Node(x)
                linked_list.add(node)
            linked_list.get_node_by_index(target_index)
        except:
            break


def fun_36():
    """
    计算字符串的编辑距离
    https://www.nowcoder.com/practice/3959837097c7413a961a135d7104c314?tpId=37&tqId=21275&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    Levenshtein 距离，又称编辑距离，指的是两个字符串之间，由一个转换成另一个所需的最少编辑操作次数。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。编辑距离的算法是首先由俄国科学家 Levenshtein 提出的，故又叫 Levenshtein Distance 。

    例如：
    字符串A: abcdefg
    字符串B: abcdef

    通过增加或是删掉字符 ”g” 的方式达到目的。这两种方案都需要一次操作。把这个操作所需要的次数定义为两个字符串的距离。
    要求：
    给定任意两个字符串，写出一个算法计算它们的编辑距离。
    """

    def lcs(a_str, b_str):
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

    while True:
        try:
            a = input()
            b = input()
            print(lcs(a, b))
        except:
            break


def fun_37():
    """
    表达式求值
    https://www.nowcoder.com/practice/9566499a2e1546c0a257e885dfdbf30d?tpId=37&tqId=21277&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    给定一个字符串描述的算术表达式，计算出结果值。
    输入字符串长度不超过 100 ，合法的字符包括 ”+, -, *, /, (, )” ， ”0-9” 。
    """
    import re
    chars = input()
    chars = re.sub(r"([-+*/()])", r" \1 ", chars)
    tmp = chars.split()
    tokens = []
    # 处理负数
    f = False
    for i, x in enumerate(tmp):
        if f == True:
            f = False
            continue
        if x == '-' and (i == 0 or tmp[i - 1] == '('):
            tokens.append(''.join(tmp[i:i + 2]))
            f = True
        else:
            tokens.append(x)

    ops = []
    vals = []

    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}

    def applyOp(v1, v2, op):
        if op == '+': return v1 + v2
        if op == '-': return v1 - v2
        if op == '*': return v1 * v2
        if op == '/': return v1 // v2

    for t in tokens:
        if t != '-' and t[0] == '-' or t.isdigit():
            vals.append(int(t))
        elif t == '(':
            ops.append(t)
        elif t == ')':
            while len(ops) > 0 and ops[-1] != '(':
                v2 = vals.pop()
                v1 = vals.pop()
                op = ops.pop()
                val = applyOp(v1, v2, op)
                vals.append(val)
            ops.pop()
        else:
            while len(ops) > 0 and precedence[ops[-1]] >= precedence[t]:
                v2 = vals.pop()
                v1 = vals.pop()
                op = ops.pop()
                val = applyOp(v1, v2, op)
                vals.append(val)
            ops.append(t)

    while len(ops) > 0:
        v2 = vals.pop()
        v1 = vals.pop()
        op = ops.pop()
        val = applyOp(v1, v2, op)
        vals.append(val)

    print(vals[-1])


def fun_38():
    """
    完全数计算
    https://www.nowcoder.com/practice/7299c12e6abb437c87ad3e712383ff84?tpId=37&tqId=21279&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    完全数（Perfect number），又称完美数或完备数，是一些特殊的自然数。
    它所有的真因子（即除了自身以外的约数）的和（即因子函数），恰好等于它本身。
    例如：28，它有约数1、2、4、7、14、28，除去它本身28外，其余5个数相加，1+2+4+7+14=28。
    """

    def is_perfect_num(n):
        res = list()
        for x in range(1, int(n / 2) + 1):
            w, y = divmod(n, x)
            if y == 0:
                if x != n:
                    res.append(x)
                if w != n:
                    res.append(w)

        if sum(set(res)) == n:
            return 1
        else:
            return -1

    while True:
        try:
            count = 0
            for x in range(1, int(input())):
                res = is_perfect_num(x)
                if res != -1:
                    count += 1
            print(count)

        except:
            break


def fun_39():
    """
    高精度整数加法
    https://www.nowcoder.com/practice/49e772ab08994a96980f9618892e55b6?tpId=37&tqId=21280&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入两个用字符串 str 表示的整数，求它们所表示的数之和。
    """

    while True:
        try:
            s1 = list(map(int, input()))[::-1]
            s2 = list(map(int, input()))[::-1]
            res = ""
            i = 0  # 遍历指针
            addd = 0  # 进位
            while i < max(len(s1), len(s2)):  # 开始遍历
                a = 0 if i >= len(s1) else s1[i]  # 获取s1中的一位数字
                b = 0 if i >= len(s2) else s2[i]  # 获取s2中的一位数字
                summ = (addd + a + b) % 10  # 计算和
                addd = (addd + a + b) // 10  # 计算进位
                res = str(summ) + res  # 组织到输出字符串中
                i += 1
            if addd > 0:  # 处理最后一位
                res = "1" + res
            print(res)  # 输出
        except:
            break


def fun_40():
    """
    查找组成一个偶数最接近的两个素数
    https://www.nowcoder.com/practice/f8538f9ae3f1484fb137789dec6eedb9?tpId=37&tqId=21283&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    任意一个偶数（大于2）都可以由2个素数组成，组成偶数的2个素数有很多种情况，本题目要求输出组成指定偶数的两个素数差值最小的素数对。
    """
    import math

    def isPrime(n):
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    while True:
        try:
            a = int(input())
            b = a // 2
            for i in range(b, 0, -1):
                if isPrime(i) and isPrime(a - i):
                    print(i)
                    print(a - i)
                    break
        except:
            break


def fun_41():
    """
    放苹果
    https://www.nowcoder.com/practice/bfd8234bb5e84be0b493656e390bdebf?tpId=37&tqId=21284&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    把m个同样的苹果放在n个同样的盘子里，允许有的盘子空着不放，问共有多少种不同的分法？
    注意：如果有7个苹果和3个盘子，（5，1，1）和（1，5，1）被视为是同一种分法。
    """
    '''
    放苹果分为两种情况，一种是有盘子为空，一种是每个盘子上都有苹果。
    令(m,n)表示将m个苹果放入n个盘子中的摆放方法总数。
    1.假设有一个盘子为空，则(m,n)问题转化为将m个苹果放在n-1个盘子上，即求得(m,n-1)即可
    2.假设所有盘子都装有苹果，则每个盘子上至少有一个苹果，即最多剩下m-n个苹果，问题转化为将m-n个苹果放到n个盘子上
    即求(m-n，n)
    '''

    def put_apple(m, n):
        if m == 0 or n == 1:
            return 1
        if n > m:
            return put_apple(m, m)
        else:
            return put_apple(m, n - 1) + put_apple(m - n, n)

    while True:
        try:
            n, m = map(int, input().split())
            print(put_apple(n, m))
        except:
            break


def fun_42():
    """
     MP3光标位置
     https://www.nowcoder.com/practice/eaf5b886bd6645dd9cfb5406f3753e15?tpId=37&tqId=21287&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
     MP3 Player因为屏幕较小，显示歌曲列表的时候每屏只能显示几首歌曲，用户要通过上下键才能浏览所有的歌曲。为了简化处理，假设每屏只能显示4首歌曲，光标初始的位置为第1首歌。
    现在要实现通过上下键控制光标移动来浏览歌曲列表，控制逻辑如下：
    歌曲总数<=4的时候，不需要翻页，只是挪动光标位置。
    光标在第一首歌曲上时，按Up键光标挪到最后一首歌曲；光标在最后一首歌曲时，按Down键光标挪到第一首歌曲。
    其他情况下用户按Up键，光标挪到上一首歌曲；用户按Down键，光标挪到下一首歌曲。
    2. 歌曲总数大于4的时候（以一共有10首歌为例）：
    特殊翻页：屏幕显示的是第一页（即显示第1 – 4首）时，光标在第一首歌曲上，用户按Up键后，屏幕要显示最后一页（即显示第7-10首歌），同时光标放到最后一首歌上。同样的，屏幕显示最后一页时，光标在最后一首歌曲上，用户按Down键，屏幕要显示第一页，光标挪到第一首歌上。
    一般翻页：屏幕显示的不是第一页时，光标在当前屏幕显示的第一首歌曲时，用户按Up键后，屏幕从当前歌曲的上一首开始显示，光标也挪到上一首歌曲。光标当前屏幕的最后一首歌时的Down键处理也类似。

    """

    def helper(cur, n, order):  # order为操作'UUUU'
        max_ = 1
        for s in order:
            # 首先是歌曲总数<=4的时候
            if s == 'U' and cur == 1:
                cur = n  # 光标在第一首歌曲上时，按Up键光标挪到最后一首歌曲
            elif s == 'U':
                cur -= 1  # 如果cur不是第一首，向上挪一首
            elif s == 'D' and cur == n:
                cur = 1  # 光标在最后一首歌曲时，按Down键光标挪到第一首歌曲
            elif s == 'D':
                cur += 1  # 如果cur不是最后一首，向下挪一首

            # 歌曲总数大于4的时候
            if n > 4:
                if cur > max_:
                    max_ = cur
                if cur < max_ - 3:
                    max_ = cur + 3
        return cur, max_

    while True:
        try:
            n, order, cur = int(input()), input(), 1  # cur为当前光标所在
            cur, max_ = helper(cur, n, order)
            ans = range(max_ - 3, max_ + 1) if n > 4 else range(1, n + 1)
            print(" ".join(map(str, ans)))
            print(cur)
        except:
            break


def fun_43():
    """
    查找两个字符串a,b中的最长公共子串
    https://www.nowcoder.com/practice/181a1a71c7574266ad07f9739f791506?tpId=37&tqId=21288&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    查找两个字符串a,b中的最长公共子串。若有多个，输出在较短串中最先出现的那个。
    注：子串的定义：将一个字符串删去前缀和后缀（也可以不删）形成的字符串。请和“子序列”的概念分开！
    """
    # 方法一
    while True:
        try:
            str1, str2 = input(), input()
            if len(str1) > len(str2):
                str1, str2 = str2, str1
            c = 0
            d = ''
            for i in range(len(str1)):
                if str1[i - c:i + 1] in str2:
                    d = str1[i - c:i + 1]
                    c += 1
            print(d)
        except:
            break

    # 方法二动态规划
    while True:
        try:
            s1 = input()
            s2 = input()
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            index, max_len = 0, 0
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        if dp[i][j] > max_len:
                            max_len = dp[i][j]
                            index = i
                    else:
                        dp[i][j] = 0
            print(s1[index - max_len:index])
        except:
            break


def fun_44():
    """
    24点游戏算法
    https://www.nowcoder.com/practice/fbc417f314f745b1978fc751a54ac8cb?tpId=37&tqId=21290&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    给出4个1-10的数字，通过加减乘除运算，得到数字为24就算胜利,除法指实数除法运算,运算符仅允许出现在两个数字之间,本题对数字选取顺序无要求，但每个数字仅允许使用一次，且需考虑括号运算
    此题允许数字重复，如3 3 4 4为合法输入，此输入一共有两个3，但是每个数字只允许使用一次，则运算过程中两个3都被选取并进行对应的计算操作。
    """

    def helper(arr, item):  # 先写一个利用递归+枚举解决算24的程序
        if item < 1:
            return False
        if len(arr) == 1:  # 递归终点，当数组arr只剩一个数的时候，判断是否等于item
            return arr[0] == item
        else:  # 如果arr不是只剩一个数，就调用函数本身（直到只剩一个为止返回真假）
            for i in range(len(arr)):
                m = arr[0:i] + arr[i + 1:]
                n = arr[i]
                if helper(m, item + n) or helper(m, item - n) or helper(m, item * n) or helper(m, item / n):
                    return True
            return False

    while True:
        try:
            if helper(list(map(int, input().split())), 24):
                print('true')
            else:
                print('false')
        except:
            break


def fun_45():
    """
    矩阵乘法
    https://www.nowcoder.com/practice/ebe941260f8c4210aa8c17e99cbc663b?tpId=37&tqId=21292&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    如果A是个x行y列的矩阵，B是个y行z列的矩阵，把A和B相乘，其结果将是另一个x行z列的矩阵C。这个矩阵的每个元素是由下面的公式决定的
    第一行包含一个正整数x，代表第一个矩阵的行数
    第二行包含一个正整数y，代表第一个矩阵的列数和第二个矩阵的行数
    第三行包含一个正整数z，代表第二个矩阵的列数
    之后x行，每行y个整数，代表第一个矩阵的值
    之后y行，每行z个整数，代表第二个矩阵的值
    """
    while True:
        try:
            x = int(input())
            y = int(input())
            z = int(input())
            A = []
            B = []
            for i in range(x):
                temp = input().split()
                temp = list(map(int, temp))
                A.append(temp)
            for j in range(y):
                temp = input().split()
                temp = list(map(int, temp))
                B.append(temp)

            res = []
            for a in range(x):
                ttt = []
                for b in range(z):
                    temp = 0
                    for c in range(y):
                        temp += A[a][c] * B[c][b]
                    ttt.append(temp)
                res.append(ttt)
            for x in res:
                out = " ".join(list(map(str, x)))
                print(out)

        except Exception as e:
            break


def fun_46():
    """
    矩阵乘法计算量估算
    https://www.nowcoder.com/practice/15e41630514445719a942e004edc0a5b?tpId=37&tqId=21293&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    例如：

    A是一个50×10的矩阵，B是10×20的矩阵，C是20×5的矩阵
    计算A*B*C有两种顺序：((AB)C)或者(A(BC))，前者需要计算15000次乘法，后者只需要3500次。
    编写程序计算不同的计算顺序需要进行的乘法次数。
    """
    # 计算式的题目，考出入栈处理括号
    # 先摘抄优秀答案：
    # 按照出入栈处理括号的思路，自己重写如下代码。
    while True:
        try:
            n = int(input())
            mdict = {}
            for i in range(n):
                mdict[chr(ord('A') + i)] = list(map(int, input().strip().split()))  ## 用strip()先去掉可能隐藏的末尾空格。再split(),防止map不过去
            s = input()
            result = 0
            temp = []
            for i in s:
                if i != ')':  # 不遇到')'就一直压栈
                    temp.append(i)
                else:  # 直接遇到')',把前两个弹出来计算乘法运算量
                    C = temp.pop()
                    B = temp.pop()
                    temp.pop()  # 弹掉前括号'('
                    result += mdict[B][0] * mdict[B][1] * mdict[C][1]
                    mdict[B] = [mdict[B][0], mdict[C][1]]  # 把当前乘积的结果存储起来
                    temp.append(B)  # 把当前乘积结果入栈
            # 因为有最外圈括号，弹完所有')'即完成整个算式的结果
            print(result)
        except:
            break


def fun_47():
    """
    字符串通配符
    https://www.nowcoder.com/practice/43072d50a6eb44d2a6c816a283b02036?tpId=37&tqId=21294&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    问题描述：在计算机中，通配符一种特殊语法，广泛应用于文件搜索、数据库、正则表达式等领域。现要求各位实现字符串通配符的算法。
    要求：
    实现如下2个通配符：
    *：匹配0个或以上的字符（注：能被*和?匹配的字符仅由英文字母和数字0到9组成，下同）
    ？：匹配1个字符
    注意：匹配时不区分大小写。
    """
    import re
    while True:
        try:
            a = input()
            b = input()
            a = a.lower()
            b = b.lower()
            a = a.replace('?', '\w{1}').replace('.', '\.').replace('*', '\w*')  # \w{1}代表匹配一个0-9且小写字母 \w*代表匹配多个0-9或小写字母
            c = re.findall(a, b)  # c中为b中与a匹配的字符串
            if b in c:
                print('true')
            else:
                print('false')
        except:
            break


def fun_48():
    """
    百钱买百鸡问题
    https://www.nowcoder.com/practice/74c493f094304ea2bda37d0dc40dc85b?tpId=37&tqId=21295&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    公元五世纪，我国古代数学家张丘建在《算经》一书中提出了“百鸡问题”：鸡翁一值钱五，鸡母一值钱三，鸡雏三值钱一。百钱买百鸡，问鸡翁、鸡母、鸡雏各几何？
    现要求你打印出所有花一百元买一百只鸡的方式。
    """
    while True:
        try:
            for i in range(20):
                for s in range(33):
                    if 5 * i + 3 * s + 1 / 3 * (100 - i - s) == 100:
                        print(i, s, 100 - i - s)
        except:
            break


def fun_49():
    """
    计算日期到天数转换
    https://www.nowcoder.com/practice/769d45d455fe40b385ba32f97e7bcded?tpId=37&tqId=21296&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    根据输入的日期，计算是这一年的第几天。
    保证年份为4位数且日期合法。
    进阶：时间复杂度：O(n)\O(n) ，空间复杂度：O(1)\O(1)
    """
    import datetime
    while True:
        try:
            year, month, day = input().split()
            input_date = datetime.date(int(year), int(month), int(day))
            today_date = datetime.date(int(year), 1, 1)
            days = (input_date - today_date).days
            print(days + 1)
        except:
            break


def fun_50():
    """
    公共子串计算
    https://www.nowcoder.com/practice/98dc82c094e043ccb7e0570e5342dd1b?tpId=37&tqId=21298&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    给定两个只包含小写字母的字符串，计算两个字符串的最大公共子串的长度。
    """
    while True:
        try:
            a = input().upper()
            b = input().upper()
            n = 0
            for i in range(len(a)):
                if a[i - n:i + 1] in b:
                    n += 1
            print(n)
        except:
            break


def fun_51():
    """
    尼科彻斯定理
    https://www.nowcoder.com/practice/dbace3a5b3c4480e86ee3277f3fe1e85?tpId=37&tqId=21299&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    验证尼科彻斯定理，即：任何一个整数m的立方都可以写成m个连续奇数之和。
    6**3 = 31+33+35+37+39+41
    """
    while True:
        try:
            num = int(input())
            while True:
                res = []
                for x in range(num * num - num + 1, num * num + num, 2):
                    res.append(x)
                if sum(res) == num ** 3:
                    res = list(map(str, res))
                    print("+".join(res))
                    break
        except:
            break


def fun_52():
    """
    火车进站
    https://www.nowcoder.com/practice/97ba57c35e9f4749826dc3befaeae109?tpId=37&tqId=21300&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    给定一个正整数N代表火车数量，0<N<10，接下来输入火车入站的序列，一共N辆火车，每辆火车以数字1-9编号，火车站只有一个方向进出，同时停靠在火车站的列车中，只有后进站的出站了，先进站的才能出站。
    要求输出所有火车出站的方案，以字典序排序输出
    """
    while True:
        try:
            n = int(input())
            trains = input().strip().split(' ')
            res = []

            def dfs(index, in_trains, out_trains):
                if trains[-1] in in_trains:
                    res.append(' '.join(out_trains + in_trains[::-1]))  # 最后一辆火车已经进站，直接输出
                    return
                elif not in_trains:
                    dfs(index + 1, in_trains + [trains[index]], out_trains)  # 还有没有已经进站的火车，只能进站
                else:
                    dfs(index, in_trains[:-1], out_trains + [in_trains[-1]])  # 当前的火车出站
                    dfs(index + 1, in_trains + [trains[index]], out_trains)  # 当前的火车进站

            dfs(0, [], [])
            res.sort()
            print('\n'.join(res))
        except:
            break


def fun_53():
    """
    将真分数分解为埃及分数
    https://www.nowcoder.com/practice/e0480b2c6aa24bfba0935ffcca3ccb7b?tpId=37&tqId=21305&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    分子为1的分数称为埃及分数。现输入一个真分数(分子比分母小的分数，叫做真分数)，请将该分数分解为埃及分数。如：8/11 = 1/2+1/5+1/55+1/110。
注：真分数指分子小于分母的分数，分子和分母有可能gcd不为1！
    :return:
    """
    while True:
        try:
            a, b = map(int, input().split('/'))
            a *= 10
            b *= 10
            res = []
            while a:
                for i in range(a, 0, -1):
                    if b % i == 0:
                        res.append('1' + '/' + str(int(b / i)))
                        a = a - i
                        break
            print('+'.join(res))
        except:
            break


def fun_54():
    """
    最长回文子串
    https://www.nowcoder.com/practice/12e081cd10ee4794a2bd70c7d68f5507?tpId=37&tqId=21308&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    给定一个仅包含小写字母的字符串，求它的最长回文子串的长度。
    :return:
    """
    while True:
        try:
            a = input()
            max_size = 1
            for i in range(0, len(a)):
                for j in range(i + 1, len(a) + 1):
                    if a[i:j] == a[i:j][::-1]:
                        if len(a[i:j]) > max_size:
                            max_size = len(a[i:j])
            print(max_size)
        except Exception as e:
            break


def fun_55():
    """
    求最大连续bit数
    https://www.nowcoder.com/practice/4b1658fd8ffb4217bc3b7e85a38cfaf2?tpId=37&tqId=21309&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    求一个int类型数字对应的二进制数字中1的最大连续数，例如3的二进制为00000011，最大连续2个1
    :return:
    """
    while True:
        try:
            x = int(input())
            byte_x = bin(x)[2:]
            list1 = sorted(list(set(byte_x.split('0'))), key=lambda x: len(x), reverse=True)
            print(len(list1[0]))
        except:
            break


def fun_56():
    """
    扑克牌大小
    https://www.nowcoder.com/practice/d290db02bacc4c40965ac31d16b1c3eb?tpId=37&tqId=21311&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    扑克牌游戏大家应该都比较熟悉了，一副牌由54张组成，含3~A、2各4张，小王1张，大王1张。牌面从小到大用如下字符和字符串表示（其中，小写joker表示小王，大写JOKER表示大王）：
    3 4 5 6 7 8 9 10 J Q K A 2 joker JOKER
    输入两手牌，两手牌之间用"-"连接，每手牌的每张牌以空格分隔，"-"两边没有空格，如：4 4 4 4-joker JOKER。
    请比较两手牌大小，输出较大的牌，如果不存在比较关系则输出ERROR。
    基本规则：
    （1）输入每手牌可能是个子、对子、顺子（连续5张）、三个、炸弹（四个）和对王中的一种，不存在其他情况，由输入保证两手牌都是合法的，顺子已经从小到大排列；
    （2）除了炸弹和对王可以和所有牌比较之外，其他类型的牌只能跟相同类型的存在比较关系（如，对子跟对子比较，三个跟三个比较），不考虑拆牌情况（如：将对子拆分成个子）；
    （3）大小规则跟大家平时了解的常见规则相同，个子、对子、三个比较牌面大小；顺子比较最小牌大小；炸弹大于前面所有的牌，炸弹之间比较牌面大小；对王是最大的牌；

    （4）输入的两手牌不会出现相等的情况。
    """
    while True:
        try:
            D = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11,
                 '2': 12, 'joker': 13, 'JOKER': 14}
            a, b = input().split('-')
            s1, s2 = a.split(), b.split()
            if a == 'joker JOKER' or b == 'joker JOKER':
                print('joker JOKER')
            elif len(s1) == len(s2):
                print(a if D[s1[0]] > D[s2[0]] else b)
            elif len(s1) == 4:
                print(a)
            elif len(s2) == 4:
                print(b)
            else:
                print('ERROR')
        except:
            break


def fun_57():
    """
     24点运算
     https://www.nowcoder.com/practice/7e124483271e4c979a82eb2956544f9d?tpId=37&tqId=21312&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
     计算24点是一种扑克牌益智游戏，随机抽出4张扑克牌，通过加(+)，减(-)，乘(*), 除(/)四种运算法则计算得到整数24，本问题中，扑克牌通过如下字符或者字符串表示，其中，小写joker表示小王，大写JOKER表示大王：

    3 4 5 6 7 8 9 10 J Q K A 2 joker JOKER

    本程序要求实现：输入4张牌，输出一个算式，算式的结果为24点。

    详细说明：

    1.运算只考虑加减乘除运算，没有阶乘等特殊运算符号，没有括号，友情提醒，整数除法要当心，是属于整除，比如2/3=0，3/2=1；
    2.牌面2~10对应的权值为2~10, J、Q、K、A权值分别为为11、12、13、1；
    3.输入4张牌为字符串形式，以一个空格隔开，首尾无空格；如果输入的4张牌中包含大小王，则输出字符串“ERROR”，表示无法运算；
    4.输出的算式格式为4张牌通过+-*/四个运算符相连，中间无空格，4张牌出现顺序任意，只要结果正确；
    5.输出算式的运算顺序从左至右，不包含括号，如1+2+3*4的结果为24，2 A 9 A不能变为(2+1)*(9-1)=24
    6.如果存在多种算式都能计算得出24，只需输出一种即可，如果无法得出24，则输出“NONE”表示无解。
    7.因为都是扑克牌，不存在单个牌为0的情况，且没有括号运算，除数(即分母)的数字不可能为0
    product('ABC', repeat=2) = AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD  返回元组，笛卡尔积，相当于嵌套的for循环
    permutations('ABCD', 2) = AB AC AD BA BC BD CA CB CD DA DB DC   长度r元组，所有可能的排列，无重复元素
    combinations('ABCD', 2) = AB AC AD BC BD CD    长度r元组，有序，无重复元素
    combinations_with_replacement('ABCD', 2)  AA AB AC AD BB BC BD CC CD DD 长度r元组，有序，元素可重复
    """

    import itertools
    while True:
        try:
            flag = 0
            D = {"J": "11", "Q": "12", "K": "13", "A": "1"}
            operators = {'0': '+', '1': '-', '2': '*', '3': '/'}

            input_cards = input().split()
            new_cards = [D.get(i, i) for i in input_cards]
            operator_cases = itertools.product(map(str, range(4)), repeat=3)  # 笛卡尔积，相当于嵌套的for循环

            for o in operator_cases:
                for i in itertools.permutations(range(4), 4):  # 长度r元组，所有可能的排列，无重复元素, combinations() 方式类似 长度r元组，有序，无重复元素
                    temp1 = '((' + new_cards[i[0]] + operators[o[0]] + new_cards[i[1]] + ')' + operators[o[1]] + new_cards[i[2]] + ')' + operators[o[2]] + new_cards[i[3]]  ## 因为运算是从前往后的，所以这里加个括号
                    temp2 = input_cards[i[0]] + operators[o[0]] + input_cards[i[1]] + operators[o[1]] + input_cards[i[2]] + operators[o[2]] + input_cards[i[3]]
                    if ('joker' in temp1) or ('JOKER' in temp1):
                        flag = 1
                        print('ERROR')
                    elif eval(temp1) == 24:
                        print(temp2)
                        flag = 2
            if flag == 0:
                print('NONE')
        except:
            break


def fun_58():
    """
    走方格的方案数
    https://www.nowcoder.com/practice/e2a22f0305eb4f2f9846e7d644dba09b?tpId=37&tqId=21314&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    请计算n*m的棋盘格子（n为横向的格子数，m为竖向的格子数）从棋盘左上角出发沿着边缘线从左上角走到右下角，总共有多少种走法，要求不能走回头路，即：只能往右和往下走，不能往左和往上走。
    """

    def f(n, m):
        if n == 0 or m == 0:
            return 1
        return f(n - 1, m) + f(n, m - 1)

    while True:
        try:
            n, m = map(int, input().split())
            print(f(n, m))
        except:
            break


def fun_59():
    """
    数组分组
    https://www.nowcoder.com/practice/9af744a3517440508dbeb297020aca86?tpId=37&tqId=21316&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入int型数组，询问该数组能否分成两组，使得两组中各元素加起来的和相等，并且，所有5的倍数必须在其中一个组中，所有3的倍数在另一个组中（不包括5的倍数），不是5的倍数也不是3的倍数能放在任意一组，可以将数组分为空数组，能满足以上条件，输出true；不满足时输出false。
    """

    def dfs(sum3, sum5, other):
        if sum3 == sum5 and len(other) == 0:
            return True
        elif len(other) == 0:
            return False
        else:
            return dfs(sum3 + other[0], sum5, other[1:]) or dfs(sum3, sum5 + other[0], other[1:])

    while True:

        try:
            n = int(input())
            num_list = list(map(int, input().split()))

            list3, list5, other = [], [], []
            for i in num_list:
                if i % 3 == 0:
                    list3.append(i)
                    continue
                if i % 5 == 0:
                    list5.append(i)
                    continue
                other.append(i)
            sum3 = sum(list3)
            sum5 = sum(list5)
            if dfs(sum3, sum5, other):
                print('true')
            else:
                print('false')


        except:
            break


def fun_60():
    """
    人民币转换
    https://www.nowcoder.com/practice/00ffd656b9604d1998e966d555005a4b?tpId=37&tqId=21318&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    考试题目和要点：

    1、中文大写金额数字前应标明“人民币”字样。中文大写金额数字应用壹、贰、叁、肆、伍、陆、柒、捌、玖、拾、佰、仟、万、亿、元、角、分、零、整等字样填写。
    2、中文大写金额数字到“元”为止的，在“元”之后，应写“整字，如532.00应写成“人民币伍佰叁拾贰元整”。在”角“和”分“后面不写”整字。
    3、阿拉伯数字中间有“0”时，中文大写要写“零”字，阿拉伯数字中间连续有几个“0”时，中文大写金额中间只写一个“零”字，如6007.14，应写成“人民币陆仟零柒元壹角肆分“。
    4、10应写作“拾”，100应写作“壹佰”。例如，1010.00应写作“人民币壹仟零拾元整”，110.00应写作“人民币壹佰拾元整”
    5、十万以上的数字接千不用加“零”，例如，30105000.00应写作“人民币叁仟零拾万伍仟元整”
    """
    while True:
        try:
            rmb = input().split('.')
            n = rmb[0]
            m = rmb[1]
            y = ["零", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖"]
            z = ["元", "拾", "佰", "仟", "万", "拾", "佰", "仟", "亿", "拾", "佰", "仟", "万亿", "拾", "佰", "仟"]
            t = ["角", "分"]

            result_b = ''
            for i in range(len(m)):
                if m[i] == '0':
                    continue
                b = y[int(m[i])] + t[i]
                result_b += b

            result_a = '人民币'
            for i in range(len(n)):
                if n[i] == '0':
                    result_a += '零'
                else:
                    a = y[int(n[i])] + z[len(n) - i - 1]
                    result_a += a
            s = result_a
            s = s.replace('零零', '零')
            s = s.replace('人民币零', '人民币')
            s = s.replace('壹拾', '拾')
            if result_b:
                print(s + result_b)
            else:
                print(s + '整')
        except:
            break


def fun_61():
    """
    字符统计
    https://www.nowcoder.com/practice/c1f9561de1e240099bdb904765da9ad0?tpId=37&tqId=21325&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    输入一个只包含小写英文字母和数字的字符串，按照不同字符统计个数由多到少输出统计结果，如果统计的个数相同，则按照ASCII码由小到大排序输出。
数据范围：字符串长度满足 1 \le len(str) \le 1000 \1≤len(str)≤1000
    """
    while True:
        try:
            temp = {}
            a = input()
            for x in a:
                temp[x] = temp.get(x, 0) + 1
            temp = list(temp.items())
            temp = sorted(temp, key=lambda y: ord(y[0]))
            temp = sorted(temp, key=lambda y: y[1], reverse=True)
            res = ""
            for x in temp:
                res += x[0]
            print(res)
        except:
            break


def fun_62():
    """
    Redraiment的走法
    https://www.nowcoder.com/practice/24e6243b9f0446b081b1d6d32f2aa3aa?tpId=37&tqId=21326&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    Redraiment是走梅花桩的高手。Redraiment可以选择任意一个起点，从前到后，但只能从低处往高处的桩子走。他希望走的步数最多，你能替Redraiment研究他最多走的步数吗？
    """
    "2 5 1 5 4 5"
    while True:
        try:
            num = int(input())
            vec = list(map(int, input().strip().split()))
            dp = [1] * num
            for i in range(1, num):
                for j in range(0, i):
                    if vec[i] > vec[j]:
                        dp[i] = max(dp[i], dp[j] + 1)
            print(max(dp))
        except:
            break


def fun_63():
    """
    求解立方根
    https://www.nowcoder.com/practice/caf35ae421194a1090c22fe223357dca?tpId=37&tqId=21330&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    计算一个浮点数的立方根，不使用库函数。
    保留一位小数。
    """
    while True:
        try:
            num = float(input())
            if num == 0:
                print(0)
            if num > 0:
                sig = 1
            else:
                sig = -1
            num = abs(num)
            if num > 1:
                start = 0
                end = num
            else:
                start = num
                end = 1
            mid = (start + end) / 2.0
            while abs(mid ** 3 - num) > 0.00001:
                if mid ** 3 < num:
                    start = mid
                else:
                    end = mid
                mid = (start + end) / 2
            print(round(sig * mid, 1))
        except:
            break


def fun_64():
    """
    求最小公倍数
    https://www.nowcoder.com/practice/22948c2cad484e0291350abad86136c3?tpId=37&tqId=21331&rp=1&ru=/ta/huawei&qru=/ta/huawei&difficulty=&judgeStatus=&tags=/question-ranking
    正整数A和正整数B 的最小公倍数是指 能被A和B整除的最小的正整数值，设计一个算法，求输入A和B的最小公倍数。
    """
    while True:
        try:
            a, b = list(map(int, input().split()))
            if a < b:
                a, b = b, a
            for i in range(a, a * b + 1, a):
                if i % b == 0:
                    print(i)
                    break
        except:
            break

# #使用-342686650:
# ret = 123456789 << 20
# print(ret)
# 得到结果129453825982464
# print(bin(ret))
# 这个二进制是11101011011110011010001010100000000000000000000
# 明显已经超出32位了
#
# 在JS上
# document.writeln(123456789 << 20);
# 得到结果是-783286272
# 这就是溢出后截取的，
#
# 在python上想实现溢出效果，找到一个函数
# #这个函数可以得到32位int溢出结果，因为python的int一旦超过宽度就会自动转为long，永远不会溢出，有的结果却需要溢出的int作为参数继续参与运算
def int_overflow(val):
    maxint = 2147483647
    flag = maxint + 1
    if not -maxint-1 <= val <= maxint:
        val = (val + flag) % (2 * flag) - flag
    return val
#
#
# ret = int_overflow(123456789 << 20)
# print(ret)
# print(bin(ret))
# 现在得到结果是-783286272
# 二进制：-101110101100000000000000000000

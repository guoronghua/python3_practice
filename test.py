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
ratings = [1,0,2]
min_val =  min(ratings)
min_val_index = ratings.index(min_val)
print(min_val_index)


def candy(ratings):
    min_val = min(ratings)
    min_val_index = ratings.index(min_val)
    i = j = min_val_index
    res = [0] * len(ratings)
    res[i] = 1
    while i > 0:
        if ratings[i - 1] > ratings[i]:
            res[i - 1] = res[i] + 1
        elif ratings[i - 1] == ratings[i]:
            if i - 2 >= 0 and ratings[i - 2] == min_val:
                res[i - 1] = 2
            else:
                res[i - 1] = 1
        else:
            res[i - 1] = max(1, res[i] - 1)
        i -= 1

    while j < len(ratings) - 1:
        if ratings[j + 1] > ratings[j]:
            res[j + 1] = res[j] + 1
        elif ratings[j + 1] == ratings[j]:
            if j + 2 < len(ratings) and ratings[j + 2] == min_val:
                res[j + 1] = 2
            else:
                res[j + 1] = 1
        else:
            res[j + 1] = max(1, res[j] - 1)
        j += 1
    print(res)
    return sum(res)

candy([1,3,2,2,1])
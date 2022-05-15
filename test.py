def swap(nums, index1, index2):
    nums[index1], nums[index2] = nums[index2], nums[index1]

def first_missing_positive(nums) -> int:
    size = len(nums)
    for i in range(size):
        # 先判断这个数字是不是索引，然后判断这个数字是不是放在了正确的地方
        # nums[i] - 1 表示这个nums[i] 应该放在的位置，即 3 应该放在索引为 2 的地方， 4 应该放在索引为 3 的地方
        while 1 <= nums[i] <= size and nums[i] != nums[nums[i] - 1]:
            #nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
            print(1, nums)
            #swap(nums, i, nums[i] - 1)
            nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
            print(2, nums)

    for i in range(size):
        if i + 1 != nums[i]:
            return i + 1

    return size + 1

# aa = first_missing_positive([3,4,-1,1])
# print(aa)

zz = [-1, 4, 3, 1]
i=1
print(zz[i])
print(zz[zz[i]-1])

aa = zz[i]-1
zz[i], zz[aa] = zz[aa], zz[i]
print(zz)
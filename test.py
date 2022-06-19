

heights = [6,7,5,2,4,5,9,3]

aa = heights[:-3]
bb = heights[3:]
print()

n = len(heights)
left, right = [0] * n, [0] * n

mono_stack = list()
for i in range(n):  # 从左往右对数组进行遍历，借助单调栈求出了每根柱子的左边界(左侧且最近的小于其高度的柱子)
    while mono_stack and heights[mono_stack[-1]] >= heights[i]:
        mono_stack.pop()
    left[i] = mono_stack[-1] if mono_stack else -1
    mono_stack.append(i)


# 输入：[6,7,5,2,4,5,9,3]
# 左侧的柱子编号分别为 [−1,0,−1,−1,3,4,5,3]
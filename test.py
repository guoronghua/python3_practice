def largestRectangleArea(heights) -> int:
    left = 0
    right = len(heights) - 1
    result = 0
    if len(heights) == 1:
        return heights[0]
    while left <= right:
        aa = heights[left:right + 1]
        bb = right - left + 1
        area = min(heights[left:right + 1]) * (right - left + 1)
        result = max([result, area])
        if heights[left] > heights[right]:
            right -= 1
        else:
            left += 1
    return result

largestRectangleArea([5,5,1,7,1,1,5,2,7,6])
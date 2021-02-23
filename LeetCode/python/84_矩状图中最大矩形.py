class Solution:
    """
    思路一：单调栈，当前元素大于栈顶元素，出栈，出栈时计算面积
    """
    def largestRectangleArea(self, heights):

        stack = []
        max_area = 0
        heights = [0] + heights + [0]
        n = len(heights)
        for i in range(n):

            if not stack or heights[i] >= heights[stack[-1]]:
                stack.append(i)
            else:
                # 采用单调栈保存最后的结果
                while stack and heights[stack[-1]] > heights[i]:
                    cur_height = heights[stack.pop()]
                    # 关键点，前面加上一个0，使得首位形式一样
                    cur_width = i - stack[-1] - 1
                    max_area = max(max_area, cur_height * cur_width)

                stack.append(i)

        return max_area
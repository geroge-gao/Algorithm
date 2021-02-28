class Solution:
    # 和84题的思路基本一样

    def largestRectangleArea(self, heights) -> int:
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

    def maximalRectangle(self, matrix) -> int:

        m = len(matrix)
        if m == 0:
            return 0
        n = len(matrix[0])
        heights = [0] * n
        ans = 0
        for i in range(m):
            for j in range(n):
                # 当前位置为0， 表示结果不会对最大面积有影响
                if matrix[i][j] == "0":
                    heights[j] = 0
                else:
                    heights[j] += 1
            ans = max(ans, self.largestRectangleArea(heights))
        return ans


if __name__ == '__main__':
    matrix = [["0", "1", "1", "0", "0"], ["1", "1", "1", "1", "1"], ["1", "0", "1", "1", "1"],
              ["1", "0", "0", "1", "0"]]

    res = Solution().maximalRectangle(matrix)
    print(res)
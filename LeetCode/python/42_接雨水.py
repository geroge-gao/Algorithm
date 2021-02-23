class Solution:
    """
    思路：最小栈
    """
    def trap(self, height) -> int:

        n = len(height)
        stack = []
        current = 0
        res = 0

        if n == 0:
            return 0

        while current < n:
            # 如果栈为空或者当前位置高度小于栈顶元素，入栈
            while stack and height[current] >= height[stack[-1]]:
                # 弹出栈顶元素
                left = stack.pop()
                if not stack:
                    break
                distance = current - stack[-1] - 1
                max_height = min(height[stack[-1]], height[current]) - height[left]
                res += max_height * distance
            stack.append(current)
            current += 1

        return res
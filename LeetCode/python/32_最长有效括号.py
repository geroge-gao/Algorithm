class Solution:
    def longestValidParentheses(self, s: str) -> int:

        stack = []
        max_len = 0
        length = 0

        if len(s) == 0:
            return 0

        for i in range(len(s)):
            if not stack or s[i] == '(' or s[stack[-1]] == ')':
                stack.append(i)
            else:
                stack.pop()
                length = i - (stack[-1] if stack else -1)

            max_len = max(max_len, length)

        return max_len

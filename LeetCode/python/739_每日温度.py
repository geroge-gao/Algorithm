class Solution:
    """
    思路一：暴力，超时
    思路二：借助栈
    """
    def dailyTemperatures(self, T):

        n = len(T)
        stack = []
        ans = []
        ans_dict = {}
        for i in range(len(T)):
            if not stack:
                stack.append(i)
            else:
                while stack:
                    j = stack[-1]
                    if T[i] > T[j]:
                        stack.pop()
                        ans_dict[j] = i - j
                    else:
                        break
                stack.append(i)

        while stack:
            i = stack.pop()
            ans_dict[i] = 0

        for i in range(n):
            ans.append(ans_dict[i])

        return ans

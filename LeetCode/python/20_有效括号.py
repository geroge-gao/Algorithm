class Solution:
    def isValid(self, s: str) -> bool:
        stack = list()
        lens = len(s)
        for i in range(lens):
            if len(stack) == 0:
                stack.append(s[i])
            else:
                if (s[i] == ')' and stack[-1] == '(') or (s[i] == '}' and stack[-1] == '{') or \
                        (s[i] == ']' and stack[-1] == '['):
                    stack.pop()
                else:
                    stack.append(s[i])

        return len(stack) == 0

if __name__ == "__main__":
    s = "()[]{}"
    res = Solution().isValid(s)
    print(res)
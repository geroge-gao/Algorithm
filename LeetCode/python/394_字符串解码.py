class Solution:
    """
    采用栈： 输入栈和输出栈
    """
    def decodeString(self, s: str) -> str:
        num_stack = []
        str_stack = []
        multi = 0
        res = ""

        for c in s:
            if '0' <= c <= '9':
                multi = multi * 10 + int(c)
            elif 'a' <= c <= 'z':
                res += c
            elif c == '[':
                num_stack.append(multi)
                str_stack.append(res)
                res = ""
                multi = 0
            else:
                current_multi = num_stack.pop()
                current_char = str_stack.pop()
                res = current_char + current_multi * res

        return res

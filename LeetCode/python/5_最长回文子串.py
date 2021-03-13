# import numpy as np
#
# class Solution:
#     def longestPalindrome(self, s):
#         str = s[::-1]
#         length = len(str)
#         matrix = [[0] * (length) for i in range(length)]
#         matrix = np.array(matrix)
#         max_len = 0
#         end = 1
#
#         for i in range(length):
#             for j in range(length):
#                 if s[i] == str[j]:
#                     if i == 0 or j == 0:
#                         matrix[i][j] = 1
#                     else:
#                         matrix[i][j] = matrix[i-1][j-1] + 1
#
#                 if matrix[i][j] > max_len:
#                     # 判断是否属于同一字符串
#                     rev_start = length - i - 1
#                     rev_end = rev_start + matrix[i][j] - 1
#                     if rev_end == j:
#                         max_len = matrix[i][j]
#                         end = i
#
#         return s[end-max_len+1:end+1]


class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        length = len(s)
        # 如果字符串长度为或者s本身是字符串，直接返回
        if length == 1 or s == s[::-1]:
            return s
        max_len, start = 1, 0
        # 遍历每一个字符，假设为回文字符的尾字符
        for i in range(1, length):
            # [i-max_len, i]，一共max_len+1个元素
            even = s[i - max_len:i + 1]
            # [i-max_len-1, i] 一共max_len+2个元素
            odd = s[i - max_len - 1:i + 1]
            if i - max_len - 1 >= 0 and odd == odd[::-1]:
                start = i - max_len - 1
                max_len += 2
                continue
            if i - max_len >= 0 and even == even[::-1]:
                start = i - max_len
                max_len += 1
                continue
        return s[start:start + max_len]


if __name__ == "__main__":
    s = 'a'
    result = Solution().longestPalindrome(s)
    print(result)




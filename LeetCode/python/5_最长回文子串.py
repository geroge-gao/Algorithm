import numpy as np

class Solution:
    def longestPalindrome(self, s: str) -> str:
        max_len = 0
        res = ''
        n = len(s)
        for i in range(2*n):
            left = int(i/2)
            right = left + i % 2
            ans = 1
            l, r = left, right
            while l >= 0 and r < n:
                if s[l] == s[r]:
                    if l == left and right == r:
                        if left == right:
                            ans = 1
                        else:
                            ans = 2
                    else:
                        ans += 2

                    left = l
                    right = r
                    l -= 1
                    r += 1
                else:
                    break

            if left >= 0 and right < n and ans > max_len:
                max_len = ans
                res = s[left:right+1]
        return res

if __name__ == "__main__":
    s = "aacabdkacaa"
    result = Solution().longestPalindrome(s)
    print(result)




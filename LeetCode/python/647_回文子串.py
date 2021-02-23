class Solution:
    """从每一个中间往两边遍历"""
    def countSubstrings(self, s: str):

        ans = 0
        n = len(s)
        for i in range(2*n):
            l = int(i/2)
            r = l + i %2
            while l >= 0 and r < n and s[l] == s[r]:
                ans += 1
                l -= 1
                r += 1

        return ans

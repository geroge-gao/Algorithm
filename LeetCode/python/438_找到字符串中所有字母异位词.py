class Solution:
    """
    思路：参考49，滑窗+哈希+排序。超时，还需要优化
    思路二：滑窗+哈希表，采用哈希将结果保存
    """
    def findAnagrams(self, s: str, p: str) :
        p_count = [0] * 26
        s_count = [0] * 26

        m = len(p)
        n = len(s)

        for c in p:
            p_count[ord(c)-97] += 1

        res = []

        left, right = 0, 0
        for right in range(n):
            if right < m - 1:
                s_count[ord(s[right]) - 97] += 1
            else:
                s_count[ord(s[right]) - 97] += 1

                if s_count == p_count:
                    res.append(left)

                s_count[ord(s[left]) - 97] -= 1
                left += 1

        return res


if __name__ == '__main__':
    s = "abab"
    p = "ab"
    res = Solution().findAnagrams(s, p)
    print(res)


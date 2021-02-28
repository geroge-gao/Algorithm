class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import defaultdict
        hash_table = defaultdict(int)
        for c in t:
            hash_table[c] += 1

        start = 0
        end = 0
        min_len = float('inf')  # 包含子串的最小长度
        count = 0  # 用于记录当前滑动窗口包含目标字符的个数，当count = len(t)，t为子串
        res = ''
        while end < len(s):
            # 当前元素在子串中，包含子串字符长度+1
            # 同时对应子串个数应该-1，目的是为了防止同一个字符重复使用
            if hash_table[s[end]] > 0:
                count += 1
            # 如果s中存在，t中不存在
            # hash_table[i]为负数
            hash_table[s[end]] -= 1
            end += 1
            while count == len(t):
                if min_len > end - start:
                    min_len = end - start
                    res = s[start: end]
                # 如果头部不在子串中，则包含子串长度-1
                if hash_table[s[start]] == 0:
                    count -= 1

                hash_table[s[start]] += 1
                start += 1

        return res


if __name__ == '__main__':
    s = "ADOBECODEBANC"
    t = "ABC"
    res = Solution().minWindow(s, t)
    print(res)

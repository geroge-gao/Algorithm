class Solution:
    # 动态规划
    def wordBreak(self, s: str, wordDict):
        word_set = set(wordDict)
        length = len(s)
        flag = [False] * (length+1)
        flag[0] = True  # 空串为合法

        for i in range(1, length+1):
            for j in range(0, i):
                if s[j: i] in word_set and flag[j]:
                    flag[i] = True

        return flag[length]




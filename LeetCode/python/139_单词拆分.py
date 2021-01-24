class Solution:
    # 动态规划
    def wordBreak(self, s: str, wordDict):
        wordSet = set(wordDict)

        length = len(s)
        flag = [False] * (length+1)
        flag[0] = True  # 空串为合法

        for i in range(1, length+1):
            for j in range(0, i):
                if s[j: i] in wordSet:
                    flag[i] = True & flag[i-j]

        return flag[length]




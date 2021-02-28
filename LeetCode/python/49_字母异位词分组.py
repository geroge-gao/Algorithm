class Solution:
    """
    思路：将字符串转换成列表排序后合并，然后进行映射，将结果相同的合并（哈希+排序）
    """
    def groupAnagrams(self, strs):

        res = {}
        for s in strs:
            word_label = ''.join(sorted(list(s)))
            if word_label not in res:
                res[word_label] = []

            res[word_label].append(s)

        return list(res.values())

if __name__ == '__main__':
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    res = Solution().groupAnagrams(strs)
    print(res)



class Solution:
    """
    常规题： hash+ 快排
    """
    def topKFrequent(self, nums, k):

        count_dict = {}
        for i in nums:
            if i not in count_dict:
                count_dict[i] = 1
            else:
                count_dict[i] += 1

        result = list(dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True)))[:k]

        return result


if __name__ == "__main__":
    data = [1,1,1,2,2,3]
    n = 2
    result = Solution().topKFrequent(data, n)
    print(result)

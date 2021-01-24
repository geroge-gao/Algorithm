class Solution:
    """
    常规题
    思路一：利用hash表，找到符合条件的
    思路二：快排，找到链表中间位置
    思路三：Boyer-Moore 投票算法
    """
    def majorityElement(self, nums):
        length = len(nums)/2
        hash_table = {}

        for i in nums:
            if i not in hash_table:
                hash_table[i] = 1
            else:
                hash_table[i] += 1

        for i, v in hash_table.items():
            if v > length:
                return i


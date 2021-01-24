class Solution:

    """
    常规题：
    思路一：堆排序
    思路二：快排
    偷个懒，直接调用函数
    """
    def findKthLargest(self, nums, k):
        return sorted(nums, reverse=True)[k-1]
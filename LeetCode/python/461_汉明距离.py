class Solution:
    """
    汉明距离，数字中有多少个1的变形
    """
    def hammingDistance(self, x, y):
        nums = x ^ y

        count = 0
        while nums != 0:
            if nums & 1:
                count += 1
            nums >>= 1

        return count

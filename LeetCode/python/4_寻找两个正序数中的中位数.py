class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        nums = []
        len1 = len(nums1)
        len2 = len(nums2)
        i = 0
        j = 0
        while i < len1 and j < len2:
            if nums1[i] < nums2[j]:
                nums.append(nums1[i])
                i += 1
            else:
                nums.append(nums2[j])
                j += 1

        if i < len1:
            nums += nums1[i: len1]
        elif j < len2:
            nums += nums2[j: len2]

        nums_size = len(nums)
        middle_index = int(nums_size/2)
        if nums_size == 0:
            return []
        elif nums_size % 2 == 0:
            return (nums[middle_index - 1] + nums[middle_index]) / 2
        else:
            return nums[middle_index]


if __name__ == "__main__":
    nums1 = [1, 2]
    nums2 = [3, 4]
    middle = Solution().findMedianSortedArrays(nums1, nums2)
    print(middle)

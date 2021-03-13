class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start_index = 0
        end_index = start_index
        max_len = 0
        n = len(s)

        while start_index < n and end_index < n:

            if end_index == 0:
                max_len = 1
            if s[end_index] not in s[start_index: end_index]:
                end_index += 1
            else:
                start_index += 1
            if end_index-start_index > max_len:
                max_len = end_index - start_index

        return max_len


if __name__ == "__main__":
    s = ''
    result = Solution().lengthOfLongestSubstring(s)

    print(result)







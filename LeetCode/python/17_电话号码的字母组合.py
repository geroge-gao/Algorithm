class Solution:
    """
    常规题，采用dfs
    """
    def letterCombinations(self, digits: str):

        num_dict = {'2': 'abc',
                    '3': 'def',
                    '4': 'ghi',
                    '5': 'jkl',
                    '6': 'mno',
                    '7': 'pqrs',
                    '8': 'tuv',
                    '9': 'wxyz'}

        res = []
        n = len(digits)

        if n == 0:
            return res

        str_list = [num_dict[i] for i in digits]

        def dfs(str_list, s, cur):
            nonlocal res
            if cur == n:
                res.append(s)
            else:
                for c in str_list[cur]:
                    s += c
                    dfs(str_list, s, cur+1)
                    s = s[:cur]

        dfs(str_list, "", 0)
        return res
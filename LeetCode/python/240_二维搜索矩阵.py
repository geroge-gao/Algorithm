import numpy as np

class Solution:
    """
    剑指offer原题：二维数组查找
    选择右上角往左下角移动，右上角和左下角的特点为
    """
    def searchMatrix(self, matrix, target):
        matrix = np.array(matrix)
        m = matrix.shape[0]
        n = matrix.shape[1]

        flag = False

        i, j = 0, n-1
        while i < m and j >= 0:
            if matrix[i][j] == target:
                flag = True
                break
            elif matrix[i][j] < target:
                i += 1
            else:
                j -= 1

        return flag
    

if __name__ == "__main__":
    data = [[-5]]
    target = -5
    result = Solution().searchMatrix(data, target)





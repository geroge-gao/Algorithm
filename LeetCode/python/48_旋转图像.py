class Solution:
    """
    其实就是要求旋转后的坐标和旋转前的坐标，由于是正方形，所以最外圈是有规律的，因此求最外圈就行了。
    """
    def rotate(self, matrix) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        mid = int(n/2)


        for i in range(mid):

            left = [a[i] for a in matrix]
            right = [b[n-i-1] for b in matrix]
            upper = matrix[i].copy()
            bottom = matrix[n-i-1].copy()

            for j in range(i, n-i):
                matrix[i][j] = left[n-j-1]
                matrix[j][n-i-1] = upper[j]
                matrix[n-i-1][j] = right[n-j-1]
                matrix[j][i] = bottom[j]
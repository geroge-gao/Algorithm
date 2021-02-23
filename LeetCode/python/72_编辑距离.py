class Solution:
    def minDistance(self, word1: str, word2: str):
        m = len(word1)
        n = len(word2)

        distance = [([0] * (n+1)) for i in range(m+1)]

        for i in range(m+1):
            distance[i][0] = i

        for i in range(n+1):
            distance[0][i] = i

        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    distance[i][j] = distance[i-1][j-1]
                else:
                    distance[i][j] = min(distance[i-1][j], distance[i][j-1], distance[i-1][j-1]) + 1

        return distance[m][n]
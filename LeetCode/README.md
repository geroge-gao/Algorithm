# LeetCode常见题型及对应解法

[TOC]



## 技巧类

矩阵数据旋转

#### [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在**[ 原地](https://baike.baidu.com/item/原地算法)** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

~~~python
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
~~~



## 哈希

1、两数之和
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数	组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

你可以按任意顺序返回答案。

    输入：nums = [2,7,11,15], target = 9
    输出：[0,1]
    解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

解析：利用哈希，对于每一个元素x，利用哈希表保存x的值及其对应的索引，然后当target-y在hash表中时，则说明前面有值x+y = target。

~~~python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = {}
        for index, val in enumerate(nums):
            if target - nums[index] in hashtable:
                return [hashtable[target - nums[index]], index]
            hashtable[nums[index]] = index

        return [0, 0]
~~~



[49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

**示例:**

```
输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```

思路：排序+哈希

~~~python
class Solution:
    """
    思路：将字符串转换成列表排序后合并，然后进行映射，将结果相同的合并
    """
    def groupAnagrams(self, strs):

        res = {}
        for s in strs:
            word_label = ''.join(sorted(list(s)))
            if word_label not in res:
                res[word_label] = []

            res[word_label].append(s)

        return list(res.values())
~~~



## 贪心算法





## 动态规划

算法解释：

动态规划（英语：Dynamic programming，简称 DP）是一种在数学、管理科学、计算机科学、经济学和生物信息学中使用的，通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。

动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。

动态规划背后的基本思想非常简单。大致上，若要解一个给定问题，我们需要解其不同部分（即子问题），再根据子问题的解以得出原问题的解。动态规划往往用于优化递归问题，例如斐波那契数列，如果运用递归的方式来求解会重复计算很多相同的子问题，利用动态规划的思想可以减少计算量。

通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，具有天然剪枝的功能，从而减少计算量：一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。

### 回文串

5.最长回文子串

给你一个字符串 `s`，找到 `s` 中最长的回文子串。 

**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

思路：动态规划

~~~python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        length = len(s)
        # 如果字符串长度为或者s本身是字符串，直接返回
        if length == 1 or s == s[::-1]:
            return s
        max_len, start = 1, 0
        # 遍历每一个字符，假设为回文字符的尾字符
        for i in range(1, length):
            # [i-max_len, i]，一共max_len+1个元素
            even = s[i - max_len:i + 1]
            # [i-max_len-1, i] 一共max_len+2个元素
            odd = s[i - max_len - 1:i + 1]
            if i - max_len - 1 >= 0 and odd == odd[::-1]:
                start = i - max_len - 1
                max_len += 2
            elif i - max_len >= 0 and even == even[::-1]:
                start = i - max_len
                max_len += 1
        return s[start:start + max_len]
~~~



### 正则表达式

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

- '.' 匹配任意单个字符
- '*' 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

根据分析其实可以得出p的第一个字符只能为'.'或者string字符，不能是'*'

正则匹配的遍历条件是两者同时匹配到终点。

​	如果两者同时到达结尾，则能匹配上，如果其中一个p已经遍历结束，但是s还没有遍历完则匹配不上

​	由于模式里面第一个不可能是'*'。判断的时候我们只需要从左往右判断就行。

​	如果p的第二字字符是*，则需要判断第一个字符是否能匹配上，如果能匹配上。则，模式有三种：

- 字符串向下移动，但是p不变。

- p移动两位，即*的中间匹配位数为0，s不动

- p不变，s向后移动。

  如果p的第二个字符不是*，判断就很简单，只需要一个一个往后面移动。

状态转移方程可以归结为
$$
f[i][j] = f[i-1][j] or f[i][j-2] or f[i-1][j-2]
$$



直接对照代码更容易理解

~~~python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i: int, j: int) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                # 如果p][j]='*'
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]
~~~

### 子序列子串系列



[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

 **示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

思路：动态规划，关键点在于当和小于0时，需要重新选择序列头

~~~python
class Solution:
    def maxSubArray(self, nums):

        cur = 0
        sum_all = -float('inf')
        for i in range(len(nums)):

            if cur > 0:
                cur += nums[i]
            else:
                cur = nums[i]

            if cur > sum_all:
                sum_all = cur

        return sum_all
~~~

[55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

示例 1：

```python
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

思路：动态规划，关键点，计算每个位置能够跳到的最远位置
$$
right = max(right, i+nums[i])
$$


~~~python
class Solution:
    """
    关键点，index+ value
    计算每个位置能够到达的最远位置
    """

    def canJump(self, nums):
        count = len(nums)
        flag = [False] * count
        flag[0] = True
        i = 0
        right_most = 0
        while i < count:
            if i <= right_most:
                right_most = max(right_most, i+nums[i])
                if right_most >= count - 1:
                    return True
            i += 1

        return False
~~~





1) 动态规划

LCS算法，将字符串倒序，然后求最长公共子串

三种状态
$$
s[i][j] = s[i-1][j-1] + 1 s[i] = s[j]
$$

$$
s[i][j] = max(s[i-1][j], s[i][j-1]), s[i]!=s[j]
$$



### 路径类问题

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

~~~python
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下

~~~

思路：动态规划

~~~python
class Solution:
    def uniquePaths(self, m, n):

        res = [(n * [0]) for i in range(m)]
        for i in range(m):
            res[i][0] = 1

        for j in range(n):
            res[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                res[i][j] = res[i][j-1] + res[i-1][j]

        return res[m-1][n-1]
~~~

[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

难度中等797收藏分享切换为英文接收动态反馈

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/11/05/minpath.jpg)

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

思路：动态规划

~~~python
class Solution:
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])

        if m == 0:
            return 0

        # 这种写法存在浅拷贝的坑
        # min_path = [[0] * n] * m

        min_path = [([0] * n) for i in range(m)]
        # 初始化第一行与第一列
        init_dis = 0
        for i in range(m):
            init_dis += grid[i][0]
            min_path[i][0] = init_dis

        init_dis = grid[0][0]
        for i in range(1, n):
            init_dis += grid[0][i]
            min_path[0][i] += init_dis

        for i in range(1, m):
            for j in range(1, n):
                min_path[i][j] = min(min_path[i-1][j], min_path[i][j-1]) + grid[i][j]

        return min_path[m-1][n-1]
~~~

### 斐波那契数列

$$
f[n] = f[n-1] + f[n-2]
$$



[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**注意：**给定 *n* 是一个正整数。

**示例 1：**

```
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
```

### 编辑距离

72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

给你两个单词 `word1` 和 `word2`，请你计算出将 `word1` 转换成 `word2` 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符 

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

思路：动态规划，关键点，空字符串是能够匹配的

~~~python
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
~~~







## 回溯

回溯算法实际上一个类似枚举的搜索尝试过程，主要是在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就 “回溯” 返回，尝试别的路径。回溯法是一种选优搜索法，按选优条件向前搜索，以达到目标。但当探索到某一步时，发现原先选择并不优或达不到目标，就退回一步重新选择，这种走不通就退回再走的技术为回溯法，而满足回溯条件的某个状态的点称为 “回溯点”。许多复杂的，规模较大的问题都可以使用回溯法，有“通用解题方法”的美称。

回溯算法的基本思想是：从一条路往前走，能进则进，不能进则退回来，换一条路再试。

一般采用dfs

[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

~~~python
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
~~~

思路：回溯，dfs

关键在于终止条件，以及如何继续想下遍历

```python
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
```


[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

示例：

~~~
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
~~~

~~~python
class Solution:
    def generateParenthesis(self, n: int):
        res = []

        def dfs(str, left, right):

            if left == 0 and right == 0:
                res.append(str)

            if right < left:
                return

            if left > 0:
                dfs(str+'(', left-1, right)
            if right > 0:
                dfs(str+')', left, right-1)
        dfs("", n, n)
        return res
~~~

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 
示例 1：

~~~python
输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]
~~~

思路：dfs，不断遍历不同的情况。

关键点：

- 终止条件
- 遍历是是否能够重复选取

~~~python
class Solution:
    def __init__(self):
        self.res_all = []

    def dfs(self, candiates, res, target):

        if sum(res) == target:
            self.res_all.append(res)
            return
        elif sum(res) > target:
            return

        for i in range(len(candiates)):
            self.dfs(candiates[i:], res + [candiates[i]], target)

    def combinationSum(self, candidates, target):

        self.dfs(candidates,[], target)

        return self.res_all
~~~



### 排列问题

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

给定一个 **没有重复** 数字的序列，返回其所有可能的全排列。

**示例:**

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```





~~~python
class Solution:
    def permute(self, nums):
        m = len(nums)
        visited = [False] * (m+1)
        res = []

        def dfs(linklist, n):
            if n == 0:
                res.append(linklist)

            for i in range(1, m+1):
                if not visited[i]:
                    visited[i] = True
                    dfs(linklist + [nums[i-1]], n - 1)
                    visited[i] = False

        dfs([], m)
        return res
~~~

[78. 子集](https://leetcode-cn.com/problems/subsets/)

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

思路：回溯

~~~python
class Solution:
    """
    第一反应是回溯
    """
    def subsets(self, nums):

        count = len(nums)
        res_all = []

        def dfs(data, target, res):

            if target == 0:
                res_all.append(res.copy())
                return

            for i in range(len(data)):
                res.append(data[i])
                dfs(data[i+1:], target-1, res)
                res.pop()

        res_all.append([])

        for i in range(1, count+1):
            dfs(nums, i, [])

        return res_all
~~~

### 迷宫求解

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例:**

```python
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true
给定 word = "SEE", 返回 true
给定 word = "ABCB", 返回 false
```

思路：

~~~python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m ,n, l = len(board), len(board[0]), len(word)
        used = [[0]*n for _ in range(m)]
        bias = [(-1,0), (1,0), (0,-1), (0, 1)]
        def dfs(c, r, location):
            nonlocal used
            flag = False
            # 判断当前字符是否匹配，若不匹配直接返回，剪枝操作
            if board[c][r]==word[location]:
                # 已匹配到最后一个字符且相同，返回True
                if location==l-1: return True
                # 当前字符字符匹配成功，后续递归过程中无法再次使用
                used[c][r] = 1
                # 对当前字符上下左右四个位置中未匹配的字符进行递归           
                for dx, dy in bias:
                    x, y = c+dx, r+dy
                    if 0<=x<m and 0<=y<n and not used[x][y]:
                        flag = flag or dfs(x, y, location+1)
                # 回溯状态返回， 此字符可再次被使用
                used[c][r] = 0
            return flag
        # 遍历二维网格中的每一个字符
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0): return True
        return False

作者：gsp_leetcode
链接：https://leetcode-cn.com/problems/word-search/solution/shen-du-you-xian-sou-suo-hui-su-xi-jie-xiang-jie-b/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
~~~



## 双指针

接水问题

11.盛水最多的容器

给你 `n` 个非负整数 `a1，a2，...，a``n`，每个数代表坐标中的一个点 `(i, ai)` 。在坐标内画 `n` 条垂直线，垂直线 `i` 的两个端点分别为 `(i, ai)` 和 `(i, 0)` 。找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

~~~
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
https://leetcode-cn.com/problems/container-with-most-water/
~~~

思路：这是一道典型的双指针题目，题目意思是利用从数组中选择两个位置，当做挡板，看中间能够容纳多少水。所以思路很简单，建立一个双指针，从左右往中间移动，每次移动左右指针中值较小的那个，每次更新的时候判断面积是否大于最大面积，如果大于，就更新。

~~~python
class Solution:
    def maxArea(self, height) -> int:

        n = len(height)
        max_area = 0
        l, r = 0, n-1

        while l < r:
            area = min(height[l], height[r]) * (r-l)
            max_area = max(area, max_area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
                
        return max_area
~~~

[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

给你一个包含 `n` 个整数的数组 `nums`，判断 `nums` 中是否存在三个元素 *a，b，c ，*使得 *a + b + c =* 0 ？请你找出所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

~~~
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
~~~

思路：排序+双指针变体

设置三个指针，首先将数组从小到大进行排序，然后从前往后一次选择一个数作为三元组的第一个数，然后设置两个左右指针，判断节点是否小于0，如果小于0，左指针往右移动，如果大于0，右指针往左移动。每一次保存上一次满足条件的结果，然后下一次满足条件时与上一次结果对比，如果不一样，保存结果。

~~~python
class Solution:
    def threeSum(self, nums):

        nums.sort()
        count = len(nums)
        result = []
        # 保留上一次结果，做去重使用
        first = -1 
        second = -1 
        third = -1
        for i in range(count-1):
            j = i + 1 # 左指针
            k = count - 1

            while j < k:
                target = nums[i] + nums[j] + nums[k]
                if target < 0:
                    j += 1
                elif target > 0:
                    k -= 1
                else:
                    # 判断是否已经存在改结果
                    if third == -1 or (nums[j] > nums[second] or nums[k] < nums[third] 
                                       or nums[i] > nums[first]):
                        # 第一次满足条件
                        result.append([nums[i], nums[j], nums[k]])
                        first = i
                        second = j
                        third = k

                    j += 1

        return result
~~~

[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

~~~
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
~~~

思路：双指针，一个先向后移动n步，然后两个指针一起移动。当前面一个移动到第n个时，后面的指针则指向第k个结点。

~~~python
def get_length(head):
    # 获取链表长度
    p = head
    count = 0
    while p is not None:
        count += 1
        p = p.next

    return count


class Solution:
    def removeNthFromEnd(self, head, n):
        m = get_length(head)

        # 返回头结点
        if m == n:
            return head.next
        k = m - n - 1
        p = head
        for i in range(k):
            p = p.next

        # 删除第n个结点
        q = p.next
        if q is not None:
            p.next = q.next
        else:
            p.next = q

        return head
~~~







[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

实现获取 **下一个排列** 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须**[ 原地 ](https://baike.baidu.com/item/原地算法)**修改，只允许使用额外常数空间。

示例：

~~~
输入：nums = [1,2,3]
输出：[1,3,2]
~~~

思路：双指针

注意到下一个排列总是比当前排列要大，除非该排列已经是最大的排列。我们希望找到一种方法，能够找到一个大于当前序列的新序列，且变大的幅度尽可能小。具体地：

我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。

从有往左遍历。找到第一个nums[i] < nums[i+1]的数，此时可以将数组分成两个部分，nums[0, i], nums[i+1, n-1]。

可以知道此时nums[i+1, n-1]此时是降序。交换nums[i]和nums[j]，然后将nums[i+1,n-1]进行升序排序就能得到结果

~~~python
class Solution:
    def nextPermutation(self, nums) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        i = len(nums) - 2
        while i >= 0:
            if nums[i] < nums[i+1]:
                break
            i -= 1

        if i < 0:
            nums.reverse()
        else:
            j = len(nums) - 1
            while j > i:
                if nums[j] > nums[i]:
                    nums[j], nums[i] = nums[i], nums[j]
                    break
                j -= 1
            left = i + 1
            right = len(nums) - 1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        return nums
~~~



[75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

难度中等802收藏分享切换为英文接收动态反馈

给定一个包含红色、白色和蓝色，一共 `n` 个元素的数组，**[原地](https://baike.baidu.com/item/原地算法)**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。 

**示例 1：**

```
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

思路：双指针

~~~python
class Solution:
    """
    思路一：sort函数
    思路二：单指针，第一次遍历，放0的位置，第二次遍历，放一的位置
    思路二：双指针，如果是0异动到前面，2移动到后面
    """
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        p, q = 0, length - 1
        i = 0
        while i <= q:
            if nums[i] == 0:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1

            if nums[i] == 2:
                nums[q], nums[i] = nums[i], nums[q]
                q -= 1

            i += 1
~~~



### 快慢指针



### 滑动窗口

​	滑动窗口问题算是双指针的一种变形

3. 无重复字符的最长子串

~~~

给定一个字符串，请你找出其中不含有重复字符的最长子串的长度。
示例 1:
输入: s = "abcabcbb"
输出: 3 
~~~

思路：设置首尾指针，都从0开始，然后尾指针end_index每向后移动一位，判断当前的字符是否在滑窗里面，如果不在

将尾指针字符加入滑窗，如果在，头指针向后移动一位，加入尾指针，然后更新滑窗的长度。如果当前长度大于最大长

，则更新最长子串。

代码：

~~~python
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
~~~

239.滑动窗口最大值

~~~
	 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值

[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

~~~

思路：

一、暴力求解，直接遍历每个元素组成的不同滑动窗口。超时

二、双端队列+滑动窗口欧

​	利用双端队列实现单调队列，每次里面保存最大的结果的索引，这样做的好处是即能取到对应的值，也可以通过index来判断当前元素距离队列头部保存index的距离，从而得到当前滑窗滑窗的大小。

~~~python
from collections import deque
class Solution:
    """
    思路一：暴力求解，直接找出方框里面最大的值，基本超时
    思路二：双端队列，其实就是利用双端队列实现最大堆，每次在结果里面保存可能最大的值
    """
    def maxSlidingWindow(self, nums, k):

        n = len(nums)

        q = deque()
        res = []
        for i in range(k):
            if not q or nums[i] <= nums[q[-1]]:
                q.append(i)
            else:
                while q and nums[i] > nums[q[-1]]:
                    q.pop()
                q.append(i)

        res.append(nums[q[0]])
        for i in range(k, n):
            while q and nums[i] > nums[q[-1]]:
                q.pop()
            q.append(i)
            if i - q[0] == k:
                q.popleft()

            res.append(nums[q[0]])
        return res
~~~

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

难度困难987收藏分享切换为英文接收动态反馈

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

**注意：**如果 `s` 中存在这样的子串，我们保证它是唯一的答案。 

**示例 1：**

```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```

思路：滑动窗口

~~~python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import defaultdict
        hash_table = defaultdict(int)
        for c in t:
            hash_table[c] += 1

        start = 0
        end = 0
        min_len = float('inf')  # 包含子串的最小长度
        count = 0  # 用于记录当前滑动窗口包含目标字符的个数，当count = len(t)，t为子串
        res = ''
        while end < len(s):
            # 当前元素在子串中，包含子串字符长度+1
            # 同时对应子串个数应该-1，目的是为了防止同一个字符重复使用
            if hash_table[s[end]] > 0:
                count += 1
            hash_table[s[end]] -= 1
            end += 1
            while count == len(t):
                if min_len > end - start:
                    min_len = end - start
                    res = s[start: end]
                # 如果头部不在子串中，则包含子串长度-1
                if hash_table[s[start]] == 0:
                    count -= 1

                hash_table[s[start]] += 1
                start += 1

        return res
~~~

类似题目：

3. 无重复字符的最长子串

4. 串联所有单词的子串

5. 最小覆盖子串

6. 至多包含两个不同字符的最长子串

7. 长度最小的子数组

8. 滑动窗口最大值

9. 字符串的排列

10. 最小区间

11. 最小窗口子序列

4、寻找两个有序数组的中位数遍

解法：

- 遍历

- 解法先保留



## 栈

### 括号问题

借助栈的特性，来判断括号是否合法

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

- 左括号必须用相同类型的右括号闭合。
- 左括号必须以正确的顺序闭合。

思路：借助辅助栈，如果当前字符是左括号，入栈。如果当前字符是右括号，出栈。如果栈空，则表明都是合法的，如果非空，表示不合法。

~~~python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = list()
        lens = len(s)
        for i in range(lens):
            if len(stack) == 0:
                stack.append(s[i])
            else:
                if (s[i] == ')' and stack[-1] == '(') or (s[i] == '}' and stack[-1] == '{') or \
                        (s[i] == ']' and stack[-1] == '['):
                    stack.pop()
                else:
                    stack.append(s[i])

        return len(stack) == 0
~~~

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

示例：

~~~
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
~~~

思路：借助栈，由于字符串只包含两种字符，因此对于左括号进展，对于右括号，出栈时，当前元素和栈顶元素的差则为有效括号的长度，不断更新有效括号，就能得到最后结果。

~~~python
class Solution:
    def longestValidParentheses(self, s: str) -> int:

        stack = []
        max_len = 0
        length = 0

        if len(s) == 0:
            return 0
        
        for i in range(len(s)):

            if not stack or s[i] == '(' or s[stack[-1]] == ')':
                stack.append(i)
            else:
                stack.pop()
                length = i - (stack[-1] if stack else -1)
            
            max_len = max(max_len, length)
        
        return max_len
~~~

[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

难度中等822收藏分享切换为英文接收动态反馈

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

**示例 1：**

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

思路：排序+栈

~~~python
class Solution:
    """
    借助栈
    """
    def merge(self, intervals):

        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

        stack = []
        for i in intervals:
            if stack and stack[-1][1] >= i[0]:
                cur = stack.pop()
                union = [min(i[0], cur[0]), max(i[1], cur[1])]
                stack.append(union)
            else:
                stack.append(i)

        return stack
~~~





### 单调栈

32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

判断入栈条件

- 如果栈空，入栈
- 当前字符为'('，入栈
- 当前栈为')'

上述三种情况无法进行匹配出栈，所以直接消除

出栈：

通过出栈时')'的下标与队列中下标进行对比。

其实有两种情况

- 当栈为空的时候，表示前面的括号都是合法的，所以合法长度为i+1
- 当栈不为空的时候，表示前面存在不合法括号，所以直接用$i- index_{not\;legal}$



[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

~~~python
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/trapping-rain-water
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
~~~

思路：单调栈



~~~python
class Solution:
    """
    思路：最小栈
    """
    def trap(self, height) -> int:

        n = len(height)
        stack = []
        current = 0
        res = 0

        if n == 0:
            return 0

        while current < n:
            # 如果栈为空或者当前位置高度小于栈顶元素，入栈
            while stack and height[current] >= height[stack[-1]]:
                # 弹出栈顶元素
                left = stack.pop()
                if not stack:
                    break
                distance = current - stack[-1] - 1
                max_height = min(height[stack[-1]], height[current]) - height[left]
                res += max_height * distance
            stack.append(current)
            current += 1

        return res
~~~

[84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

难度困难1215收藏分享切换为英文接收动态反馈

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram.png)

以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)

图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

**示例:**

```
输入: [2,1,5,6,2,3]
输出: 10
```

思路：

~~~python
class Solution:
    """
    思路一：单调栈，当前元素大于栈顶元素，出栈，出栈时计算面积
    """
    def largestRectangleArea(self, heights):

        stack = []
        max_area = 0
        heights = [0] + heights + [0]
        n = len(heights)
        for i in range(n):

            if not stack or heights[i] >= heights[stack[-1]]:
                stack.append(i)
            else:
                # 采用单调栈保存最后的结果
                while stack and heights[stack[-1]] > heights[i]:
                    cur_height = heights[stack.pop()]
                    # 关键点，前面加上一个0，使得首位形式一样
                    cur_width = i - stack[-1] - 1
                    max_area = max(max_area, cur_height * cur_width)
                stack.append(i)

        return max_area
~~~

#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/09/14/maximal.jpg)

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```

思路：和前面两题一样



## 树

### 树的遍历



1、先序中序，后序遍历二叉树

[96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

给定一个整数 *n*，求以 1 ... *n* 为节点组成的二叉搜索树有多少种？

：

解析：二叉搜索数的定义是，所有对于所有节点都有

- 根节点大于左节点的
- 根节点小于右节点
- 先序遍历二叉搜索树，得到有序单调递增序列

定义两个函数

- $G(n)$: 长度为n的序列能构成不同的二叉搜索树的个数
- $F(i,n)$：以$i$为根、序列长度为n的不同二叉搜索树的个数

以不同的i作为节点，可得到
$$
G(n)=\sum_{i=1}^nF(i,n)
$$
边界条件：G(0)=G(1)=1

$F(i, n)$的定点固定，所以能够组成的二叉搜索树个数是由两个子节点决定的。即

[1, i]和[i+1, n]决定。而[i+1, n]等价于[1, n-i-1]

所以有$F(i, n)=G(i)G(n-i)$

所以最后的递推公式可以转换成
$$
G(n)=\sum_{i=1}^nG(i)G(n-i)
$$
该公式可以归纳为卡塔兰数
$$
C(n+1)=\frac{2(2n+1)}{n+2}C(n)
$$


~~~python
def numTrees(self, n):

    G = [0] * (n+1)
    G[0] = G[1] = 1

    for i in range(2, n+1):
        for j in range(1, i+1):
            G[i] += G[j-1] * G[i-j]

    return G[n]
~~~



## 链表

### 

~~~
2. 两数相加（链表）
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
~~~

思路：链表相加，简单题目

### 链表合并

[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

~~~python
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
~~~

思路：链表基操

~~~python
class Solution:
    def mergeTwoLists(self, l1, l2):
        head = ListNode(0, None)
        node = head
        p = l1
        q = l2
        while p is not None and q is not None:
            if p.val <= q.val:
                node.next = p
                node = p
                p = p.next
            else:
                node.next = q
                node = q
                q = q.next

        if p is not None:
            node.next = p
        if q is not None:
            node.next = q
        return head.next
~~~



[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

#### 未完待续





148.[排序链表](https://leetcode-cn.com/problems/sort-list/)

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 

归并排序



## 并查集



## 查找算法

### 二分查找

4.寻找两个正序数组的中位数

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

 **示例 1：**

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

思路一：将两个序列进行合并，类似链表的操作方式，然后找到中位数。

思路二：二分查找。暂时没看懂

~~~python
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

~~~

二分查找解法：

#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的索引，否则返回 `-1` 。

思路：二分查找，比较简单，直接贴代码。

~~~python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[len(nums) - 1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
~~~



## 归纳推理



 

## 排序算法

在[计算机科学](https://zh.wikipedia.org/wiki/計算機科學)与[数学](https://zh.wikipedia.org/wiki/數學)中，一个**排序算法**（英语：Sorting algorithm）是一种能将一串资料依照特定排序方式进行排列的一种[算法](https://zh.wikipedia.org/wiki/算法)。最常用到的排序方式是数值顺序以及[字典顺序](https://zh.wikipedia.org/wiki/字典順序)。有效的排序算法在一些算法（例如[搜索算法](https://zh.wikipedia.org/wiki/搜尋算法)与[合并算法](https://zh.wikipedia.org/w/index.php?title=合併算法&action=edit&redlink=1)）中是重要的，如此这些算法才能得到正确解答。排序算法也用在处理文字资料以及产生人类可读的输出结果。基本上，排序算法的输出必须遵守下列两个原则：

1. 输出结果为递增序列（递增是针对所需的排序顺序而言）
2. 输出结果是原输入的一种[排列](https://zh.wikipedia.org/wiki/排列)、或是重组

### 归并排序

### 快速排序

### 选择排序

### 冒泡排序

### 堆排序

### 桶排序

### 基数排序

### 希尔排序
















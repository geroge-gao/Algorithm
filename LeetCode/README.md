# LeetCode常见题型及对应解法

## 哈希

1、两数之和

C++map函数，注意map和unorderd_map的区别

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;
        for(int i=0; i<nums.size(); i++)
        {
            if(hashtable.find(target - nums[i])!= hashtable.end())
                return {hashtable.find(target - nums[i])->second, i};
            hashtable[nums[i]] = i;
        }
        return {};
    }
};
```



## 动态规划

### 背包问题

## 回溯







## 双指针

#### 滑动窗口

一系列整理的题目

3、无重复字符串



https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/solution/hua-dong-chuang-kou-by-powcai/



4、寻找两个有序数组的中位数遍

解法：

- 遍历

- 解法先保留

5、最长回文字符串

1) 动态规划

LCS算法，将字符串倒序，然后求最长公共子串

三种状态
$$
s[i][j] = s[i-1][j-1] + 1 s[i] = s[j]
$$

$$
s[i][j] = max(s[i-1][j], s[i][j-1]), s[i]!=s[j]
$$



## 栈

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





## 树

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

[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。



148.[排序链表](https://leetcode-cn.com/problems/sort-list/)

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 

归并排序



## 排序算法












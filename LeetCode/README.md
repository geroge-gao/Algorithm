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

LCS算法，将字符串倒序，然后求最长公共子串











# 
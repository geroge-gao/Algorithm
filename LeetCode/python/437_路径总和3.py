class Solution:
    """
    前缀和+递归，与560题思路类似，注意
    """
    def pathSum(self, root: TreeNode, sum: int) -> int:

        count = 0
        prefix = {0: 1}

        def dfs(root, total, prefix):
            nonlocal count, sum

            if not root:
                return
            else:
                total += root.val
                if total - sum in prefix:
                    count += prefix[total - sum]

                if total in prefix:
                    prefix[total] += 1
                else:
                    prefix[total] = 1
                dfs(root.left, total, prefix)
                dfs(root.right, total, prefix)
                # 保持前缀都是在一条路径下面
                prefix[total] -= 1

        dfs(root, 0, prefix)
        return count

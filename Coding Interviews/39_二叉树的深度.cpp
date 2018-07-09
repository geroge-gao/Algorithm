
//求二叉树的深度
int TreeDepth(TreeNode* pRoot)
{
    int ldepth,rdepth;
    if(pRoot==NULL)
        return 0;
    ldepth = TreeDepth(pRoot->left)+1;
    rdepth = TreeDepth(pRoot->right)+1;
    return ldepth > rdepth?ldepth:rdepth;    
}

/*
判断是否为平衡二叉树，平衡二叉树要求每一个结点的高度差不超过1
*/

bool IsBalanced(TreeNode *pRoot)
{
    if(pRoot==NULL)
        return true;

    int left=TreeDepth(pRoot->left);
    int right=TreeDepth(pRoot->right);
    int diff=left-right;
    if(diff>1||diff<-1)
        return false;

    return IsBalanced(pRoot->left)&&IsBalanced(pRoot->right);
}
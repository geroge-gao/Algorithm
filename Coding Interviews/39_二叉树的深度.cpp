
int TreeDepth(TreeNode* pRoot)
{
    int ldepth,rdepth;
    if(pRoot==NULL)
        return 0;
    ldepth = TreeDepth(pRoot->left)+1;
    rdepth = TreeDepth(pRoot->right)+1;
    return ldepth > rdepth?ldepth:rdepth;
    
}
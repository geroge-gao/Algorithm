
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};

void ConvertNode(TreeNode *pNode,TreeNode **pLastNodeinList)
{
    if(pNode)
    {
        TreeNode *pCurrent=pNode;
        if(pCurrent->left!=NULL)
            ConvertNode(pCurrent->left,pLastNodeinList);
        pCurrent->left=*pLastNodeinList;
        if(*pLastNodeinList!=NULL)
            (*pLastNodeinList)->right=pCurrent;
        *pLastNodeinList=pCurrent;
        if(pCurrent->right!=NULL)
            ConvertNode(pCurrent->right,pLastNodeinList);
    }
}

TreeNode* Convert(TreeNode* pRootOfTree)
{
    TreeNode *pLastNodeinList=NULL;
    ConvertNode(pRootOfTree,&pLastNodeinList);
    TreeNode *pHeadofList=pLastNodeinList;
    //将指针移到表头
    while(pHeadofList!=NULL&&pHeadofList->left!=NULL)
        pHeadofList=pHeadofList->left;
    return pHeadofList;
}
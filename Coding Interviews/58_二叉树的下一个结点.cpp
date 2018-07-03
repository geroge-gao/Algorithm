TreeLinkNode* GetNext(TreeLinkNode* pNode)
{
    if(pNode==NULL)
        return pNode;
    TreeLinkNode * pNext;
    if(pNode->right!=NULL)
    {
        TreeLinkNode *pRight=pNode->right;
        while(pRight->left!=NULL)
            pRight=pRight->left;

        pNext=pRight;
    }
    else if(pNode->next!=NULL)
    {
        TreeLinkNode *pCurrent=pNode;
        TreeLinkNode *pRoot=pNode->next;//得到父节点
        while(pRoot!=NULL&&pCurrent==pRoot->right)
        {
            pCurrent=pRoot;
            pRoot=pRoot->next;
        }
        pNext=pRoot;
    }

    return pNext;
}

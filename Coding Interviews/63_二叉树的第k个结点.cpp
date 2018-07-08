int count=0;
void KthNodeCore(TreeNode* root,int k,TreeNode* &p)
{
    if(root)
    {
        KthNodeCore(root->left,k,p);
        count++;
        if(count==k)
            p=root;
        KthNodeCore(root->right,k,p);
    }
}

TreeNode* KthNode(TreeNode* pRoot, int k)
{
    TreeNode *p=NULL;
    if(pRoot==NULL||k==0)
        return NULL;
    KthNodeCore(pRoot,k,p);
    return p;
}
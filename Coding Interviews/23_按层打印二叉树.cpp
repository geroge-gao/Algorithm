vector<int> PrintFromTopToBottom(TreeNode* root) 
{
    vector<int> v;
    queue<TreeNode*> q;
    TreeNode *t;
    if(root==NULL)
        return v;
    q.push(root);
    
    while(!q.empty())
    {
        t=q.front();
        v.push_back(t->val);
        q.pop();
        if(t->left!=NULL)
            q.push(t->left);
        if(t->right!=NULL)
            q.push(t->right);
    }
    return v;
}
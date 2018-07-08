vector<vector<int> > res;
vector<int> path;

void Preorder(TreeNode *root,int target,int cur)
{
    if(root)
    {
        path.push_back(root->val);
        cur+=root->val;
        if(root->left==NULL&&root->right==NULL)
            if(target==cur)
                res.push_back(path);
        Preorder(root->left,target,cur);
        Preorder(root->right,target,cur);
        path.pop_back();
    }
}

vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
    int current=0;
    Preorder(root,expectNumber,current);
    return res;
}
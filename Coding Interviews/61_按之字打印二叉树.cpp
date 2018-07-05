/*
author:gerogegao
date:2018.7.5
*/

vector<vector<int> > Print(TreeNode* pRoot)
{
    vector<int> level;
    vector<vector<int> > array;
    if(pRoot==NULL)
        return array;
    queue<TreeNode*> s;
    s.push(pRoot);
    int nextLevel=0;//统计每一层的个数
    int toBePrinted=1;

    while(!s.empty())
    {
        TreeNode *p=s.front();
        level.push_back(p->val);
        if(p->left)
        {
            s.push(p->left);
            nextLevel++;
        }
        if(p->right)
        {
            s.push(p->right);
            nextLevel++;
        }
        s.pop();
        toBePrinted--;
        if(toBePrinted==0)
        {
            toBePrinted=nextLevel;
            array.push_back(level);
            nextLevel=0;
            level.clear();
        }
    }
    return array;
}

struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};

void Mirror(TreeNode *pRoot) 
{
    TreeNode *temp;
    if(pRoot!=NULL)
    {
        temp=pRoot->left;
        pRoot->left=pRoot->right;
        pRoot->right = temp;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
}
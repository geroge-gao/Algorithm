链接：https://www.nowcoder.com/questionTerminal/8a19cbe657394eeaac2f6ea9b0f6fcf6
来源：牛客网

        int in_size = in.size();
 
        if(in_size == 0)
 
            return NULL;
 
        vector<int> pre_left, pre_right, in_left, in_right;
 
        int val = pre[0];
 
        TreeNode* node = new TreeNode(val);//root node is the first element in pre
 
        int p = 0;
 
        for(p; p < in_size; ++p){
 
            if(in[p] == val) //Find the root position in in 
 
                break;
 
        }
 
        for(int i = 0; i < in_size; ++i){
 
            if(i < p){
 
                in_left.push_back(in[i]);//Construct the left pre and in 
 
                pre_left.push_back(pre[i+1]);
 
            }
 
            else if(i > p){
 
                in_right.push_back(in[i]);//Construct the right pre and in 
 
                pre_right.push_back(pre[i]);
 
            }
 
        }
 
        node->left = reConstructBinaryTree(pre_left, in_left);
 
        node->right = reConstructBinaryTree(pre_right, in_right);
 
        return node;
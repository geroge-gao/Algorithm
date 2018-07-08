/*
author:gerogegao
date:2018/7/7
*/

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};



void Serialize(TreeNode *root,string &str)
{
    if(root==NULL)
        str+='#';
    else
}

/*
12#36###75##4##
*/

char* Serialize(TreeNode *root)
{
    string str;
    if(!root)
        return NULL;

    Serialize(root,str);

    cout<<"str:"<<str<<endl;
    int len=str.size();
    char *a=new char[len+1];
    for(int i=0;i<len;i++)
        a[i]=str[i];
    a[len]='\0';
    return a;
}

int num=0;

void Deserialize(TreeNode* &root,char *str)
{
    if(*(str+num)=='\0')
        return ;

    if(*(str+num)=='#')
        root=NULL;
    else
    {
        root=(TreeNode*)malloc(sizeof(TreeNode));
        root->val=*(str+num);
        ++num;
        Deserialize(root->left,str);
        ++num;
        Deserialize(root->right,str);
    }
}

TreeNode* Deserialize(char *str)
{
    TreeNode *p;
    Deserialize(p,str);
    return p;
}    {
        str+=root->val;
        Serialize(root->left,str);
        Serialize(root->right,str);
    }

}

/*
12#36###75##4##
*/

char* Serialize(TreeNode *root)
{
    string str;
    if(!root)
        return NULL;

    Serialize(root,str);

    cout<<"str:"<<str<<endl;
    int len=str.size();
    char *a=new char[len+1];
    for(int i=0;i<len;i++)
        a[i]=str[i];
    a[len]='\0';
    return a;
}

int num=0;

void Deserialize(TreeNode* &root,char *str)
{
    if(*(str+num)=='\0')
        return ;

    if(*(str+num)=='#')
        root=NULL;
    else
    {
        root=(TreeNode*)malloc(sizeof(TreeNode));
        root->val=*(str+num);
        ++num;
        Deserialize(root->left,str);
        ++num;
        Deserialize(root->right,str);
    }
}

TreeNode* Deserialize(char *str)
{
    TreeNode *p;
    Deserialize(p,str);
    return p;
}
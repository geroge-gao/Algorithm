/*
输入数据：
8
1 2 4 7 3 5 6 8
4 7 2 1 5 3 8 6

*/


#include<stdio.h>
#include<stdlib.h>

typedef struct BinaryTree
{
    int  data;
    struct BinaryTree *left;
    struct BinaryTree *right;
}BitNode,*Bitree;

BinaryTree * ConstructureCore(int *startPreorder,int *endPreorder,int *startInorder,int *endInorder)
{
	//先序遍历的第一个数组为根节点
	int rootValue=startPreorder[0];
	Bitree root=(Bitree)malloc(sizeof(BitNode));
	root->data=rootValue;
	root->left=NULL;
	root->right=NULL;
	if(startPreorder == endPreorder)
	{
		//子树最后只有一个结点时
		if(startInorder == endInorder && *startPreorder == *startInorder)
			return root;
		else
			throw "Invalid input";
	}
	//在中序遍历中寻找根节点
	int *rootInorder=startInorder;
	while(rootInorder!=endInorder&&*rootInorder!=rootValue)
		rootInorder++;

	if(rootInorder == endInorder && *rootInorder != rootValue)
		throw "Invalid input.";
	//求左子树的长度
	int leftLength=rootInorder - startInorder;
	//在先序遍历中求左子树的最后结点位置
	int *leftPreorderEnd = startPreorder+leftLength;
	if(leftLength>0)
	{
		//构建左子树
		root->left=ConstructureCore(startPreorder+1,leftPreorderEnd,startInorder,rootInorder-1);
	}
	if(leftLength< endPreorder- startPreorder)
	{
		root->right=ConstructureCore(leftPreorderEnd+1,endPreorder,rootInorder+1,endInorder);
	}
	return root;
}

Bitree Construct(int *preorder,int *inorder,int length)
{
	if(preorder==NULL||inorder==NULL||length<=0)
		return NULL;
	return ConstructureCore(preorder,preorder+length-1,inorder,inorder+length-1);
}

void PostVisit(Bitree bt)
{
	if(bt)
	{
		PostVisit(bt->left);
		PostVisit(bt->right);
		printf("%d ",bt->data);
	}
}

int main()
{
	int a[10],b[10];
	int length;
	Bitree root;
	scanf("%d",&length);
	for(int i=0;i<length;i++)
		scanf("%d",&a[i]);
	for(int i=0;i<length;i++)
		scanf("%d",&b[i]);
	root=Construct(a,b,length);
	PostVisit(root);
    return 0;
}

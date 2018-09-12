#include<iostream>
#include<string>
using namespace std;

int Min(int a,int b,int c)
{
	int t=0;
	t=a>b?b:a;
	return t>c?c:t;
}

/*
动态规划判断条件
如果s[i]=s[j]，那么不需要插入或者改变所以有
distance[i][j]=distance[i-1][j-1];
如果s[i]!=s[j]
分为删除，修改和增加三种情况
当为删除情况时有
distance[i][j]=distance[i-1][j]+1
当为增加情况时
distance[i][j]=distance[i-1][j]+2
当为改变的情况
distance[i][j]=distance[i-1][j-1]+1

*/

int edit_distance(string s1,string s2)
{
	int l1=s1.length();
	int l2=s2.length();
	int **distance;      
	distance=new int*[l1+1];
	for(int i=0;i<=l1;i++)
		distance[i]=new int[l2+1];

	//初始化两条边界
	for(int i=0;i<=l1;i++)
		distance[i][0]=i;
	for(int i=0;i<=l2;i++)
		distance[0][i]=i;

	for(int i=1;i<=l1;i++)
		for(int j=1;j<=l2;j++)
		{
			if(s1[i]==s2[j])
			{
				distance[i][j]=distance[i-1][j-1];
			}
			else
			{
				distance[i][j]=Min(distance[i][j-1],distance[i-1][j],distance[i-1][j-1])+1;
			}
		}
	return distance[l1][l2];
}

int main()
{
	string s1,s2;
	cin>>s1>>s2;
	int dis=edit_distance(s1,s2);
	cout<<dis<<endl;
	return 0;
}
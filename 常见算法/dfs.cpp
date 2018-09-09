#include<iostream>
#include<vector>
using namespace std;

int flag[10000]={0};
int num=0;

int FirstArc(vector<vector<int> > v,int p)
{
	for(unsigned int i=0;i<v[p].size();i++)
		if(!flag[i]&&i!=p&&v[p][i])
			return i;
	return -1;
}

int NextArc(vector<vector<int >> v,int p,int j)
{
	for(unsigned  i=j;j>=0&&i<v.size();i++)
		if(!flag[i]&&v[p][i])
			return i;
	return -1;
}

void dfs(vector<vector<int >> v,int p,int count)
{

	flag[p]=1;
	if(count==v.size())
	{
		num++;
		return;
	}	
	
	for(int unsigned i=FirstArc(v,p);i>=0&&i<v.size();i=NextArc(v,p,i))
	{
		if(!flag[i])
		{
			dfs(v,i,count+1);
		}
	}
}

void InitFlag(int n)
{
	for(int i=0;i<n;i++)
		flag[i]=0;
}

void dfs_traverse(vector<vector<int >> v)
{
	for(unsigned int i=0;i<v.size();i++)
	{		
		InitFlag(v.size());
		dfs(v,i,1);
	}
}

int main()
{	
	int n,m;
    cin>>n>>m;
	vector<vector<int >> v;
	v.resize(n);
	for(unsigned int i=0;i<v.size();i++)
	{
		v[i].resize(n);
	}

	for(unsigned int i=0;i<v.size();i++)
		for(unsigned int j=0;j<v.size();j++)
		{
			v[i][j]=0;
		}


	int a,b;
	for(int i=0;i<m;i++)
	{
		cin>>a>>b;
		v[a-1][b-1]=1;
	}
	
	dfs_traverse(v);
	cout<<num<<endl;

	system("pause");
    return 0;
}

/*
3
3
1 2 2 1 2 3
1
*/
 
static bool compare(int a,int b)
{
  return a>b; //升序排列，如果改为return a>b，则为降序
}

vector<int> multiply(const vector<int>& A)
{
    vector<int> B,C,D;
    int num=1;
    int len=A.size();
    //计算出前面一半的数组
    for(int i=0;i<len;i++)
    {
        if(i==0)
        {
            num=1;
            C.push_back(num);
        }
        else
        {
            num*=A[i-1];
            C.push_back(num);
        }
    }

    //计算出后面一半的数组
    for(int i=len-1;i>=0;i--)
    {
        if(i==len-1)
        {
            num=1;
            D.push_back(num);
        }
        else
        {
            num*=A[i+1];
            D.push_back(num);
        }
    }

    reverse(D.begin(),D.end());

    for(int i=0;i<len;i++)
        B.push_back(C[i]*D[i]);
    return B;
}
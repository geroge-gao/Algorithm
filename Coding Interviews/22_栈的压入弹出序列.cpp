bool IsPopOrder(vector<int> pushV,vector<int> popV) 
{
    int len=pushV.size();
    int t=popV.size();
    stack<int> s;
    bool flag=false;
    int i=0,j=0;

    //边界判断
    if(len==0||t==0||len!=t)
        return false;
    for(int i=0;i<len;i++)
    {
        s.push(pushV[i]);
        while(!s.empty()&&s.top()==popV[j])
        {
            s.pop();
            j++;
        }
    }
    return s.empty();
}
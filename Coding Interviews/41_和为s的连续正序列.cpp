//两个数的和为s

vector<int> FindNumbersWithSum(vector<int> data,int sum)
{
    vector<int> num;
    int len=data.size();
    if(len==0)
        return num;

    int left=0;
    int right=len-1;

    while(right>left)
    {
        int Curr=data[left]+data[right];
        if(Curr>sum)
            right--;
        else if(Curr<sum)
            left++;
        else
        {
            num.push_back(data[left]);
            num.push_back(data[right]);
            break;
        }
    }

    return num;
}



//和为s的连续正序列
void GetSeq(vector<vector<int> > &res,int start,int end)
{
    vector<int> temp;
    for(int i=start;i<=end;i++)
        temp.push_back(i);
    res.push_back(temp);
}

vector<vector<int> > FindContinuousSequence(int sum)
{
    vector<vector<int> > seq;
    vector<int> temp;

    if(sum<3)
        return seq;

    int small=1;
    int big=2;
    int middle=(sum+1)/2;
    int curSum=small+big;
    int count=0;

    while(small<middle)
    {
        if(curSum<sum)
        {
            big++;
            curSum+=big;
        }
        else if(curSum>sum)
        {
            curSum-=small;
            small++;
        }
        else
        {
            GetSeq(seq,small,big);
            big++;
            curSum+=big;
        }
    }

    return seq;
}
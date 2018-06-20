//利用归并排序的思想

long long InversePairsCore(vector<int> &data,vector<int> &copy,int start,int end)
{
    if(start==end)
    {
        copy[start]=data[start];
        return 0;
    }

    int length=(end-start)/2;
    long long left = InversePairsCore(copy,data,start,start+length);
    long long right = InversePairsCore(copy,data,start+length+1,end);

    int i = start+length;
    int j=end;
    int index = end;
    long long count=0;
    while(i>=start&&j>=start+length+1)
    {
        if(data[i]>data[j])
        {
            copy[index--]=data[i--];
            count+=j-start-length;
        }
        else
        {
            copy[index--]=data[j--];
        }
    }

    for(;i>=start;--i)
        copy[index--]=data[i];
    for(;j>=start+length+1;--j)
        copy[index--]=data[j];

        return left+right+count;
}

int InversePairs(vector<int> data)
{
    if(data.size()==0)
        return 0;

    vector<int> copy;

    for(int i=0;i<data.size();i++)
        copy.push_back(data[i]);

    long long count=InversePairsCore(data,copy,0,data.size()-1);
    copy.clear();
    return count%1000000007;
}
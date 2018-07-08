//采取双端队列解决这个问题。

//直接采用暴力法解决
int Max(vector<int> num,int start,int size)
{
    int max=num[start];
    for(int i=start;i<start+size;i++)
        if(num[i]>max)
        {
            max=num[i];
        }

    return max;
}

vector<int> maxInWindows(const vector<int>& num, unsigned int size)
{
    vector<int> s;
    for(int i=0;i<=num.size()-size;i++)
    {
        int max=Max(num,i,size);
        s.push_back(max);
    }
    return s;
}

//采用辅助数据结构双端队列
vector<int> maxInWindows(const vector<int>& num, unsigned int size)
{
    vector<int> maxWindows;
    if(num.size()>=size&&size>=1)
    {
        deque<int> index;
        for(unsigned int i=0;i<size;i++)
        {
            while(!index.empty()&&num[i]>=num[index.back()])
                index.pop_back();
            index.push_back(i);
        }

        for(unsigned int i=size;i<num.size();i++)
        {
            maxWindows.push_back(num[index.front()]);
            while(!index.empty()&&num[i]>=num[index.back()])
                index.pop_back();
            if(!index.empty()&&index.front()<=i-size)
                index.pop_front();

            index.push_back(i);
        }
        maxWindows.push_back(num[index.front()]);
    }
    return maxWindows;
}
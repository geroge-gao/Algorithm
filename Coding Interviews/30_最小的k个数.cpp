int Partition(vector<int> &A,int low,int high)
{
    int t=A[low];
    while(low<high)
    {
        while(A[high]>=t&&low<high)
            high--;
        A[low]=A[high];
        while(A[low]<t&&low<high)
            low++;
        A[high]=A[low];
    }
    A[low]=t;
    return low;
}

vector<int> GetLeastNumbers(vector<int> input,int k)
{
    vector<int> output;
    if(input.size()==0)
        return output;
    int start=0;
    int end=input.size()-1;
    int index=Partition(input,start,end);
    while(index!=k-1)
    {
        if(index>k-1)
        {
            end=index-1;
            index=Partition(input,start,end);
        }
        else
        {
            start=index+1;
            index=Partition(input,start,end);
        }
    }

    for(int i=0;i<k;i++)
        output.push_back(input[i]);

    return output;
}

//适合海量数据的方法

typedef multiset<int,greater<int> > intSet;
typedef multiset<int,greater<int> >::iterator setIterator;

void GetLeastNumbers(const vector<int> &data, intSet & leastNumbers,int k)
{
    leastNumbers.clear();

    if(k<1||data.size()<k)
        return ;
    vector<int>::const_iterator iter=data.begin();

    for(;iter!=data.end();++iter)
    {
        if(leastNumbers.size()<k)
            leastNumbers.insert(*iter);
        else
        {
            setIterator iterGreatest = leastNumbers.begin();
            if(*iter<*(leastNumbers.begin()))
            {
                leastNumbers.erase(iterGreatest);
                leastNumbers.insert(*iter);
            }
        }
    }
}
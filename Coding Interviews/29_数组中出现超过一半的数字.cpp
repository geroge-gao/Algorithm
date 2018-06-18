int Partition(int *A,int low,int high)
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


bool CheckInvalidArray(int *number,int length)
{
    bool g_bInputInvalid=false;
    if(number<=0||number==NULL)
        g_bInputInvalid=true;
    return g_bInputInvalid;
}

bool CheckMoreThanHalf(int *numbers,int length,int number)
{
    int times=0;
    for(int i=0;i<length;i++)
        if(numbers[i]==number)
            times++;
    bool isMoreThanHalf=true;
    if(times*2<=length)
        isMoreThanHalf=false;
    return isMoreThanHalf;
}

int MoreThanHalfNum(int *numbers,int length)
{
    if(CheckInvalidArray(numbers,length))
        return 0;

    //除法的效率更高
    int middle=length>>1;
    int start=0;
    int end=length-1;
    int index=Partition(numbers,start,end);
    while(index!=middle)
    {
        if(index>middle)
        {
            end=index-1;
            index=Partition(numbers,start,end);
        }
        else
        {
            start=index+1;
            index=Partition(numbers,start,end);
        }
    }
    int result=numbers[middle];
    if(!CheckMoreThanHalf(numbers,length,result))
        result=0;

    return result;

}
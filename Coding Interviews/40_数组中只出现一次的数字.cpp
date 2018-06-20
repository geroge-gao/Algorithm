
unsigned int FindFirstBitIs1(int num)
{
    unsigned int indexBit=0;
    while(((num&1)==0)&&(indexBit<8*sizeof(int)))
    {
        num=num>>1;
        ++indexBit;
    }
    return indexBit;
}

bool IsBit1(int num,unsigned int indexBit)
{
    num=num>>indexBit;
    return num&1;
}

void FindNumsAppearOnce(vector<int> data,int *num1,int *num2)
{
    int length=data.size();
    if(data.size()<2)
        return;

    int result = 0;
    for(int i=0;i<length;i++)
        result^=data[i];

    //找到第一个不为1的数
    unsigned int indexOf1=FindFirstBitIs1(result);
    *num1=*num2=0;

    for(int j=0;j<length;j++)
    {
        if(IsBit1(data[j],indexOf1))
            *num1^=data[j];
        else
            *num2^=data[j];
    }
}
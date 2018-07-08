bool IsContinuous( vector<int> numbers )
{
    int len=numbers.size();
    if(len==0)
        return false;
    sort(numbers.begin(),numbers.end());

    int numberofZero = 0;
    int numberofGrap = 0;

    //统计数组中0的个数
    for(int i=0;i<len;i++)
        if(numbers[i]==0)
            numberofZero++;

    int small=numberofZero;
    int big=small+1;

    while(big<len)
    {
        if(numbers[small]==numbers[big])
            return false;
        numberofGrap+=numbers[big]-numbers[small]-1;
        small=big;
        big++;
    }
    return numberofZero>=numberofGrap;
}
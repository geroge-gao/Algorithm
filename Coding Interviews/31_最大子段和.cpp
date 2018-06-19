int FindGreatestSumOfSubArray(vector<int> array)
{
    int sum=INT_MIN,b=0;

    for(int i=0;i<array.size();i++)
    {
        if(b>0)
            b+=array[i];
        else
            b=array[i];
        if(b>=sum)
            sum=b;
    }
    return sum;
}
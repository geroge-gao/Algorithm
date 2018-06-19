
int NumberOf1(unsigned int n)
{
    int number=0;
    while(n)
    {
        if(n%10==1)
            number++;
        n/=10;
    }
    return number;
}

int NumberOf1Between1AndN_Solution(int n)
{
    int number=0;
    for(int i=1;i<=n;i++)
        number+=NumberOf1(i);
    return number;
}
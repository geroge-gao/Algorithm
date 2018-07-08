
//递归
int Fibonacci(int n)
{
	if(n==1)
		return 1;
	if(n<=0)
		return 0;
	return f(n-1)+f(n-1);
}

//非递归

int Fibonacci(int n) 
{
    int f[40];
    f[0]=0,f[1]=1;
    for(int i=2;i<40;i++)
        f[i]=f[i-1]+f[i-2];
    return f[n];
}



//跳台阶
int jumpFloor(int number) 
{
    int *f=new int[number+1];
    f[1]=1;
    f[2]=2;
    for(int i=3;i<=number;i++)
        f[i]=f[i-1]+f[i-2];
    return f[number];
}

//变态跳台阶
int jumpFloorII(int number) 
{
    int *sum=new int[number+1];
    sum[1]=1;
    sum[2]=2;
    for(int i=3;i<=number;i++)
    {
        sum[i]=0;
        for(int j=1;j<i;j++)
            sum[i]+=sum[j];
        sum[i]+=1;
    }
    return sum[number];
}

//矩阵覆盖
int rectCover(int number) 
{
    if(number == 0)
        return 0;
    if(number == 1)
        return 1;
    int a=1,b=2;
    int c=b;
    for(int i=3;i<=number;i++)
    {
        c=a+b;
        a=b;
        b=c;
    }
    return c;
}


//暴力求解，没有考虑到异常

int numberOf1(int n)
{
	int count=0;
	whille(n)
	{
		if(n&1)
			count++；
		n=n>>1;
	}
}

//改进版

int numberOf1(int n)
{
	int count=0;
	unsigned int flag=1;
	while(flag)
	{
		if(n&flag)
			count++;
		flag=flag<<1;
	}
	return count;
}
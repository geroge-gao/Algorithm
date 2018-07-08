
bool flag = true;

bool equal(double a,double b)
{
    if(abs(a-b)<0.000001)
        return true;
    return false;
}

double Power(double base, int exponent) 
{
    bool flag=true;
    if(equal(base,0.0)&&exponent<0)
    {
        flag = false;
        return 0;
    }
    unsigned int abs_exponent=abs(exponent);
    double result = 1;
    for(int i=1;i<=abs_exponent;i++)
        result*=base;
    if(exponent<0)
        result = 1/result;        
    return result;
}

    
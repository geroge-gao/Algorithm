int FirstNotRepeatingChar(string str)
{
    const int size=128;
    int a[size]={0};

    for(int i=0;i<str.size();i++)
        a[str[i]]++;
    for(int i=0;i<str.size();i++)
        if(a[str[i]]==1)
            return i;
    return -1;
}
//左旋转字符串

string LeftRotateString(string str, int n)
{
    int length = str.length();
    string temp;

    int i;
    for(i=n;i<length;i++)
        temp[i-n]=str[i];
    for(int j=0;j<n;j++)
    {
        temp[i-n]=str[j];
        i++;
    }
    for(int j=0;j<length;j++)
        str[j]=temp[j];
    return str;
}

//翻转单词顺序列
string ReverseSentence(string str)
{
    int len=str.length();
    int count=0;
    //将整个数组进行翻转,用到了内置函数
    reverse(str.begin(),str.end());
    cout<<str<<endl;
    int front=0,back=0;
    while(front<len&&back<len)
    {
        while(str[front]==' ')
            front++;
        back=front;
        while(str[back]!=' '&&back<len)
            back++;
        reverse(str.begin()+front,str.begin()+back);
        front=back+1;
    }
    return str;
}
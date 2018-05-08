#include <iostream>
#include<stdio.h>
using namespace std;

void replaceSpace(char *str,int length)
{
        int i=0,count=0;
        int p1,p2;
        if(length==0||str==NULL)
            return ;
        while(str[i]!='\0')
        {
            if(str[i]==' ')
                count++;
            i++;
        }
        int len=i+count*2;
        if(len>length)
            return;
        p1=i-1;
        p2=len-1;
        str[len]='\0';
        while(p1>=0&&p1!=p2)
        {
            if(str[p1]!=' ')
            {
                str[p2]=str[p1];
                p2--;
            }
            else
            {
                str[p2--]='0';
                str[p2--]='2';
                str[p2--]='%';
            }
            p1--;
        }
}

//思考题，插入数据

void Insert(int a[],int b[],int l1,int l2)
{
    int p1,p2;
    int len=l1+l2;
    p1=l1-1;
    p2=l2-1;
    int c=len-1;
    while(p2>=0)
    {
        if(a[p1]>=b[p2])
        {
            a[c]=a[p1];
            c--;
            p1--;
        }
        else
        {
            a[c]=b[p2];
            c--;
            p2--;
        }
    }

    for(int i=0;i<len;i++)
        cout<<a[i]<<" ";

}



int main()
{
//    char a[20];
//    gets(a);
//    replaceSpace(a,20);
//    cout<<a<<endl;
    int a[30],b[10];
    for(int i=0;i<6;i++)
        cin>>a[i];
    for(int i=0;i<3;i++)
        cin>>b[i];

    Insert(a,b,6,3);
    return 0;
}

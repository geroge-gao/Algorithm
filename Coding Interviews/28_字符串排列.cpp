#include<iostream>

using namespace std;

void Permutation(char *pStr,char *pBegin)
{
    if(pStr==NULL)
        return ;

        if(*pBegin=='\0')
            cout<<pStr<<endl;
        else
        {
            for(char *pCh=pBegin;*pCh!='\0';pCh++)
            {
                char temp=*pCh;
                *pCh=*pBegin;
                *pBegin=temp;

                Permutation(pStr,pBegin+1);

                temp=*pCh;
                *pCh=*pBegin;
                *pBegin=temp;
            }
        }
}

int main()
{
    char a[]="abc";
    Permutation(a,a);
    return 0;
}
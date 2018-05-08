#include <iostream>
#include<stack>
#include<stdio.h>
#include<stdlib.h>
using namespace std;

struct ListNode {
        int val;
        struct ListNode *next;
};

void CreateList(ListNode *l,int n)
{
    int m;
    ListNode *p,*rear;
    rear=l;
    for(int i=0;i<n-1;i++)
    {
        scanf("%d",&m);
        p=(ListNode*)malloc(sizeof(ListNode));
        p->val=m;
        p->next=rear->next;
        rear=p;
        rear=p;
    }
}

void printListFromTailToHead(ListNode* head)
{
    if(head!=NULL)
    {
        printListFromTailToHead(head->next);
    }
    printf("%d ",head->val);
}

void printListFromTail(ListNode *l)
{
    stack<int> s;
    while(l!=NULL)
    {
        s.push(l->val);
        l=l->next;
    }
    while(s.empty()==true)
    {
        int b=s.top();
        s.pop();
        printf("%d ",b);
    }
}

int main()
{
    int m,n;
    ListNode *l=(ListNode*)malloc(sizeof(ListNode));

    scanf("%d",&n);
    scanf("%d",&m);
    l->val=m;
    l->next=NULL;
    CreateList(l->next,n);
    // printListFromTailToHead(l);
    printListFromTail(l);
    return 0;
}
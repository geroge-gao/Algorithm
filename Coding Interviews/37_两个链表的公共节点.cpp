
//首先得到链表的长度差，然后将其对应到同一标准下。

int GetListLength(ListNode *pHead)
{
    ListNode *pnode=pHead;
    int length=0;
    while(pnode!=NULL)
    {
        length++;
        pnode=pnode->next;
    }
    return length;
}

ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2)
{
    int len1=GetListLength(pHead1);
    int len2=GetListLength(pHead2);

    int diff=fabs(len1-len2);

    int i=0;
    if(len1>len2)
        while(i<diff)
        {
            pHead1=pHead1->next;
            i++;
        }
    else
        while(i<diff)
        {
            pHead2=pHead2->next;
            i++;
        }
    while(pHead1&&pHead2&&pHead1!=pHead2)
    {
        pHead1=pHead1->next;
        pHead2=pHead2->next;
    }
    return pHead1;
}
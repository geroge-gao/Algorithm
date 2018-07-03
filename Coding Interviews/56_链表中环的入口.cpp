ListNode* EntryNodeOfLoop(ListNode* pHead)
{
    ListNode *pFast,*pSlow;
    pFast=pSlow=pHead;
    while(pFast!=NULL&&pFast->next!=NULL)
    {
        pFast=pFast->next->next;
        pSlow=pSlow->next;
        if(pFast==pSlow)
            break;
    }

    if(pFast==NULL||pFast->next==NULL)
        return NULL;

    pFast=pHead;
    while(pFast!=pSlow)
    {
        pFast=pFast->next;
        pSlow=pSlow->next;
    }

    return pFast;
}
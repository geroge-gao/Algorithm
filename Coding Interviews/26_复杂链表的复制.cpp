void CloneNodes(RandomListNode *pHead)
{
    RandomListNode *pNode = pHead;
    while(pNode!=NULL)
    {
        RandomListNode *pClone=new RandomListNode(0);
        pClone->label=pNode->label;
        pClone->next=pNode->next;
        pClone->random=NULL;
        pNode->next=pClone;
        pNode=pClone->next;
    }
}

//复制random指针
void CloneRandom(RandomListNode *pHead)
{
    RandomListNode *pNode=pHead;

    while(pNode!=NULL)
    {
        RandomListNode *pClone=pNode->next;
        if(pNode->random!=NULL)
            pClone->random=pNode->random->next;
        pNode=pClone->next;
    }
}

RandomListNode* ReconnectNodes(RandomListNode* pHead)
{
    RandomListNode *pNode=pHead;
    RandomListNode *pClone=NULL;
    RandomListNode *pCHead=NULL;
    if(pNode!=NULL)
    {
        pCHead=pClone=pNode->next;
        pNode->next=pClone->next;
        pNode=pNode->next;
    }
    while(pNode!=NULL)
    {
        pClone->next=pNode->next;
        pClone=pClone->next;
        pNode->next=pClone->next;
        pNode=pNode->next;
    }
    return pCHead;
}

RandomListNode* Clone(RandomListNode *pHead)
{
    CloneNodes(pHead);
    CloneRandom(pHead);
    return ReconnectNodes(pHead);
}
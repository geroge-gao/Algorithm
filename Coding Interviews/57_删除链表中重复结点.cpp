ListNode* deleteDuplication(ListNode* pHead)
{
    ListNode* head=new ListNode(0);
    ListNode* p=head;
    ListNode* q=pHead;
    while(q){
        while(q!=NULL&&q->next!=NULL&&q->next->val==q->val){
            int tmp=q->val;
            while(q!=NULL&&q->val==tmp)
                q=q->next;
        }
        p->next=q;
        p=p->next;
        if(q)
            q=q->next;
    }   
    return head->next;
}
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode *m,*p,*q;
        if(pHead1==NULL)
            return pHead2;
        else if(pHead2==NULL)
            return pHead1;
        if(pHead1->val<pHead2->val)
        {
            m=pHead1;
            pHead1=pHead1->next;
        }
        else
        {
            m=pHead2;
            pHead2=pHead2->next;
        }
        q=m;
        while(pHead1&&pHead2)
        {
            if(pHead1->val<pHead2->val)
            {
                q->next=pHead1;
                q=pHead1;
                pHead1=pHead1->next;
            }
            else
            {
                q->next=pHead2;
                q=pHead2;
                pHead2=pHead2->next;
            }
        }
        while(pHead1)
        {
            q->next=pHead1;
            q=q->next;
            pHead1=pHead1->next;
        }
        while(pHead2)
        {
            q->next=pHead2;
            q=q->next;
            pHead2=pHead2->next;
        }
        return m;
    }
};

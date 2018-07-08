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
    ListNode* ReverseList(ListNode* pHead) {
        ListNode* p=pHead;
        ListNode *temp,*q;
        if(p==NULL||p->next==NULL)
            return p;
        q=p->next;
        p->next=NULL;
        while(q->next!=NULL)
        {
            temp=q->next;
            q->next=p;
            p=q;
            q=temp;
        }
        q->next=p;
        return q;
    }
};
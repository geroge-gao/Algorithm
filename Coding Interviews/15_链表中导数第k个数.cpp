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
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode *p = pListHead;
        ListNode *q;
        int len=0;
        //判断k是否大于链表长度
        for(ListNode *m=p;m!=NULL;m=m->next)
            len++;
        
        if(p==NULL||k==0||k>len)
            return NULL;
        
        for(unsigned int i=0;i<k-1;i++)
            p=p->next;
        for(q=pListHead;p->next!=NULL;p=p->next)
            q=q->next;
        return q;
    }
};
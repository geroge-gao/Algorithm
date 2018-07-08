class Solution {
public:
    void Insert(int num)
    {
        if(p.empty()||num>=p.top())
            p.push(num);
        else
            q.push(num);
        int len=q.size()-p.size();
        if(len==1||len==2)
        {
            int t=q.top();
            q.pop();
            p.push(t);
        }
        else if(len==-2)
        {
            int t=p.top();
            p.pop();
            q.push(t);
        }
    }

    double GetMedian()
    {
        int len=q.size()+p.size();
        if(len%2)
            return p.top();
        else
        {
            int a=p.top();
            int b=q.top();
            double c=(double)(a+b)/2;
            return c;
        }
    }
private:
    priority_queue<int, vector<int>, greater<int> > p;  // 小顶堆
    priority_queue<int, vector<int>, less<int> > q;     // 大顶堆
};
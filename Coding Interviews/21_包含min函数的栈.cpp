class Solution {
public:
    void push(int value) {
        s.push(value);
        if(b.size()==0)
            b.push(value);
        else if(value<b.top())
            b.push(value);
    }

    void pop() {
        int k1=s.top();
        int k2=b.top();
        s.pop();
        if(k1==k2)
            b.pop();
            
    }

    int top() 
    {
        return s.top();
    }

    int min() 
    {
        return b.top();
    }
private:
    stack<int> s;
    stack<int> b;
};
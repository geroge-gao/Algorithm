#include <iostream>
#include <stack>
using namespace std;

#define MAXSIZE 20

class Queue
{
public:
    Queue();
    ~Queue();
    void push(int i);
    int pop();
private:
    stack<int> s1;
    stack<int> s2;
};

Queue::Queue()
{

}

Queue::~Queue()
{

}

void Queue::push(int i)
{
    s1.push(i);
}

int Queue::pop()
{
    int t;
    if(s2.empty()==true)
    {
        while(s1.empty()==false)
        {
            t=s1.top();
            cout<<t<<endl;
            s1.pop();
            s2.push(t);
        }
    }

    if(s2.size()==0)
    {
        cout<<"queue is empty\n"<<endl;
        return -1;
    }

    t=s2.top();
    s2.pop();
    return t;
}



int main(int argc, char const *argv[])
{
    int m;
    Queue q;
    for(int i=0;i<4;i++)
    {
        cin>>m;
        q.push(m);
    }

    for(int i=0;i<4;i++)
    {
        m=q.pop();
        cout<<m<<" ";
    }

    return 0;
}

class Solution {
public:

    Solution()
    {
        N++;
        sum+=N;
    }

    static void reset()
    {
        N=0;
        sum=0;
    }

    static unsigned getSum()
    {
        return sum;
    }

    int Sum_Solution(int n)
    {
        Solution::reset();

        Solution *s=new Solution[n];
        delete []s;
        s=NULL;
        return Solution::getSum();
    }

private:
    static unsigned int N;
    static unsigned int sum;
};

unsigned int Solution::N=0;
unsigned int Solution::sum=0;
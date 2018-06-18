class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int len = rotateArray.size();
        int p1=0,p2=len-1;
        while(p1+1!=p2)
        {
            int mid=(p1+p2)/2;
            if(rotateArray[mid]<rotateArray[p1])
                p2=mid;
            else
                p1=mid;
        }
        return rotateArray[p2];
                    
    }
};
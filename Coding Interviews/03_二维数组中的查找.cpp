bool Find(int target, vector<vector<int> > array) 
{
        bool flag=false;
        int rows=array.size();
        int colnums=array[0].size();
        int i=0,j=colnums-1;
        while(i<rows&&j>=0)
        {
            if(array[i][j]<target)
                i++;
            else if(array[i][j]>target)
                j--;
            else if(array[i][j]==target)
            {
                flag=true;
                break;
            }
        }
        return flag;
}

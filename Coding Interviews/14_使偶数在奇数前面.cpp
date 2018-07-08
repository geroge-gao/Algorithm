void reOrderArray(vector<int> &array) 
{
    vector<int> temp;
    int len=array.size();
    int count=0;
    for(int i=0;i<len;i++)
        if(array[i]%2==1)
            temp.push_back(array[i]);
    
    for(int i=0;i<len;i++)
        if(array[i]%2==0)
            temp.push_back(array[i]);
    array=temp;
}
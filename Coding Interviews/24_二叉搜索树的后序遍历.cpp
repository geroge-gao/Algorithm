bool VerifySquenceOfBST(vector<int> sequence) 
{
    int mid=0,root;
    bool lflag=true,rflag=true;
    vector<int> left,rights;

    if(sequence.size()==0)
        return false;
    int len=sequence.size();
    root=sequence[len-1];
    for(int i=0;i<len-1;i++)
        if(sequence[i]<root)
            mid=i;
        else
            break;

    for(int i=mid+1;i<len-1;i++)
        if(sequence[i]<root)
            return false;

    for(int i=0;i<mid;i++)
        left.push_back(sequence[i]);
    for(int i=mid+1;i<len-1;i++)
        rights.push_back(sequence[i]);

    if(left.size()>0)
        lflag = VerifySquenceOfBST(left);
    if(rights.size()>0)
        rflag = VerifySquenceOfBST(rights);
    return lflag&&rflag;
}
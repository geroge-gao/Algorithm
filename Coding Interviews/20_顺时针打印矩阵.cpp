vector<int> printMatrix(vector<vector<int> > matrix) 
{        
    int m=matrix.size();
    int n=matrix[0].size();
    vector<int> b;

    int left=0,right=n-1,top=0,bottom=m-1;
    if(m==0||n==0)
        return b;

    while(left<=right&&top<=bottom)
    {
        for(int i=left;i<=right;i++)//从上面开始赋值
            b.push_back(matrix[top][i]);
        
        for(int i=top+1;i<=bottom;i++)//从右边开始赋值
            b.push_back(matrix[i][right]);
        
        for(int i=right-1;i>=left&&top!=bottom;i--)//从下面开始赋值
            b.push_back(matrix[bottom][i]);
        
        for(int i=bottom-1;i>=top+1&&left!=right;i--)
            b.push_back(matrix[i][left]);
        
        top++;
        left++;
        right--;
        bottom--;
    }
    return b;
}
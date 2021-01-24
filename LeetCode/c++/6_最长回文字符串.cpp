class Solution {
public:
    string longestPalindrome(string s) {
        string str = s;
        reverse(str.begin(), str.end());
        int length = s.size();

        vector<int> tmpVec(length, 0);
        vector<vector<int>> matrix(length, tmpVec);

        int max_len = 0;
        int end=0;

        for(int i=0; i<length; i++)
            for(int j=0; j<length; j++)
            {
                if(s[i] == str[j])
                    if(i==0||j==0)
                        matrix[i][j]=1;
                    else
                        matrix[i][j]=matrix[i-1][j-1]+1;

                if (matrix[i][j] > max_len)
                {
                    //判断比较的字符串是不是来源自同一个字符串
                    int preJ = length - 1 - j;
                    int nowJ = preJ + matrix[i][j] - 1;
                    if (nowJ == i)
                    {
                        end = i;
                        max_len = matrix[i][j];
                    }
                }

            }

        return s.substr(end-max_len+1, max_len);

    }
};

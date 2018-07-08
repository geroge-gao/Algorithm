string s;
char hash[256] = {0};
//Insert one char from stringstream
void Insert(char ch)
{
    s += ch;
    hash[ch]++;
}
//return the first appearence once char in current stringstream
char FirstAppearingOnce()
{
    int size = s.size();
    for(int i = 0; i < size; i++){
        if(hash[s[i]] == 1)
            return s[i];
    }
    return '#';
}
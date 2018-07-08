bool matchCore(char *str,char *pattern)
{
    if(*str=='\0'&&*pattern=='\0')
        return true;
    if(*str!='\0'&&*pattern=='\0')
        return false;

    if(*(pattern+1)=='*')
    {
        if(*pattern==*str||(*pattern=='.'&&*str!='\0'))
            return matchCore(str+1,pattern)||matchCore(str,pattern+2)||matchCore(str+1,pattern+2);
        else//如果第一个不匹配，直接忽略第一个
            return matchCore(str,pattern+2);
    }
    if(*str==*pattern||(*pattern=='.'&&*str!='\0'))
        return matchCore(str+1,pattern+1);

    return false;
}

bool match(char* str, char* pattern)
{
    if(str==NULL||pattern==NULL)
        return false;
    return matchCore(str,pattern);
}
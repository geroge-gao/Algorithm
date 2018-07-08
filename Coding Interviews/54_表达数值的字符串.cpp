void ScanDigits(char **string)
{
    while(**string!='\0'&&**string>='0'&&**string<='9')
        (*string)++;
}
//传递参数是需要将地址传入。
bool isExponetial(char **string)
{
    if(**string!='e'&&**string!='E')
        return false;

    ++(*string);
    if(**string=='+'||**string=='-')
        ++(*string);

    if(**string=='\0')
        return false;
    ScanDigits(string);
    return **string=='\0'?true:false;
}

bool isNumeric(char* string)
{
    if(string==NULL)
        return false;

    if(*string=='+'||*string=='-')
        ++string;

    if(*string=='\0')
        return false;

    bool numeric=true;

    ScanDigits(&string);
    if(*string!='\0')
    {
        if(*string=='.')
        {
            ++string;
            ScanDigits(&string);
            if(*string=='e'||*string=='E')
                numeric=isExponetial(&string);
        }
        else if(*string=='E'||*string=='e')
            numeric=isExponetial(&string);
        else//出现其他字符
            numeric=false;
    }
    return numeric&&*string=='\0';
}
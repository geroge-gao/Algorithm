# 剑指offer思路总结

题目都能够在牛客网上AC

## 面试题3 二维数组中查找

先选右上角一次向左下移动

## 面试题4 替换空格

思路：先遍历一遍统计字符串长度并且统计空格个数，然后增添相对应的格式，从后向前将空格移动。

## 面试题5 从末到头打印链表

思路一：利用递归遍历链表，最后打印。

思路二：利用栈实现。

## 面试题6 重建二叉树
题目：

    给定二叉树的前序遍历和中序遍历的结果，重建该二叉树。例如给定前序遍历为{1,2,4,7,3,5,6,8},中序遍历序列为{4,7,2,1,5,3,8,6}.

思路：

根据先序中序遍历的特点可知,

- 对于先序遍历，其第一个访问的为根节点因此根节点为1，而对于中序遍历而言，先左子树然后是根节点，最后是右子树。因此{4,7,2}为左子树，{5,4,8,6}为右子树。对应于先序{2,4,7},{3,5,6,8}，一次可以类推,2为左子树的根节点，3为右子树的根节点，最后递归求解就能够得到整棵树。

## 面试题7 用两个栈实现队列

思路：

很简单，利用两个栈，先将数据压入其中一个栈，然后将数据一次弹出来压入另外一个栈。这样原来在栈底的数据在另一个栈中就变到了栈顶的位置。但是需要注意的是如果弹出的栈不为空，就不能再往弹出栈添加元素。

## 面试题8 旋转数组的最小数字

旋转数组指的是将一个有序数组进行旋转，例如[3,4,5,1,2]是[1,2,3,4,5]的旋转数组。

思路：

采用二分查找的思想。根据旋转数组的特点，可以将旋转数组分为[3,4,5],[1,2]。因此要求最小值为1。像二分查找一样设置两个指针分别指向数组头部和尾部。p1指向3，p2指向2。然后mid=(p1+p2)/2。有mid指向5，此此时p1=mid,然后mid=(p1+p2)/2=4，此时此时p2<mid。所以p2=mid。此时p2=p1+1。结束判断条件。

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


​    


## 面试题9 斐波那契数列

斐波拉契数列

    f(1)=f(2)=1;
    f(n)=f(n-1)+f(n-2);//n>=3;

### 青蛙跳台阶问题
青蛙每次可以调1或2块台阶，问调n块台阶的跳发。

这个问题其实也是一个斐波拉契数列问题，对于一层台阶只有一层跳法，对于两层台阶，可以直接跳两格，也可以从第一层挑一格。

所以递推公式为：

    f(1)=1;
    f(2)=2;
    f(3)=f(1)+f(2);

### 变态跳台阶问题

青蛙可以调1级，也可以跳2级。。。，也可以跳n级，问跳n级台阶的跳法。

思路：跳第一层有一种跳法，条第二层有两种跳法，当跳第三层的时候，可以直接跳也可以从第一层和第二层跳。因此关系式为：

f(1)=1;

f(2)=2;

f(3)=f(1)+f(2)+1;

f(n)=$\sum_{i=1}^{n-1}f(i)+1=2^{n-1}$

### 矩阵覆盖

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft2h890tmmj30ef03u74h.jpg)

用$2\times1$的的矩阵覆盖$2\times8$的种类

其实这也是一个斐波拉契数列。

先拼2 * 1和 2 * 2，再在此基础上拼凑。

所以递推公式为f(n)=f(n-1)+f(n-1)

## 面试题10 二进制中1的个数

##  面试题11 数值的整数次方

关键在于异常处理，冗余性处理。

## 面试题14 调整顺序使奇数位于偶数前面

采取一个辅助数组，遍历原始数组，如果值为奇数，就加入到辅助数组当中。代码很简单，跟高效算法可参见剑指offer。

## 面试题15 链表中倒数第k个结点

思路很简单，利用双指针设置游标。将第一个指针移动k步，然后将第二个指针指向表头，同时移动最后当第一个指针指向表尾的时候，第二个指针则是指的导数第k个结点。

易犯错误，没有考虑为空的情况，或者k大于表的长度。

## 面试题16 反转链表

借助辅助结点将链表next指针反转即可。

## 面试题17 合并两个排序链表

严版数据结构基础题。

## 面试题18 树的子结构

要判断是树A是否包含于树B，方法很简单，首先判断A的根节点是否在树B中，如果在B中，找到该节点，然后从该节点开始按照同样的遍历方式遍历树B和树A，如果每次结点值都相等，则包含，否则，不包含。

## 面试题19 二叉树的镜像

先序遍历，将二叉树左右子节点交换，注意不是交换左右子节点的值。

## 面试题20 顺时针打印矩阵

直接上代码，不废话。

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

## 面试题21 包含min函数的栈

增加一个辅助栈，存储当前栈中最小的数。当辅助栈为空时候直接数据同时压入两个栈，当不为空的时候，首先将数据压入数据栈中，然后判断当前数据是否比辅助栈中的数据的值小，如果小，直接压入。这样就能保证每次辅助栈栈顶都是数据栈的最小值。当弹出数据的时候首先判断两个栈顶的数据是否相等，如果相等，就将两个栈栈顶数据都弹出。如果不相等，只需要弹出数据栈栈顶。

## 面试题22 栈的压入和弹出序列

给出栈的输入序列，判断输出序列是否合法。

对于输入的两个序列，直接创建一个站，将输入序列按照顺序输入，然后压入栈，根据输出的对比，看是否能够得到输出序列。例如例如输入序列为{1,2,3,4,5}，输出序列为为{4,5,3,2,1}.

思路，首先压入1不等于4，继续压入2,3,4。输出序列首位为4所以弹出4，输出序列指针指向5，并且压入输入序列中的5,发现相等继续弹出，此时栈中压入了{1,2,3}。然后栈顶为3，输出指针指向3，继续弹出。以此类推，最后发现栈为空。则表明输出为合法。

## 面试题23 从上往下打印二叉树

借助队列。

## 面试题24 二叉搜索树的后序遍历序列

二叉排序树的特点为左子树结点小于根节点，右子树序列大于根节点。并且序列最后一位是根节点。因此首先找到根节点，然后从头到尾遍历数组，找到小于根节点的连续序列，即为左子树，大于根节点的即为右子树，递归遍历，直到得到最后结果。如果不满足上述条件返回false。

## 面试题25 二叉树中和为某一值的路径

回溯法

## 面试题26 复杂链表的复制

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft2pl1p0zcj30d10530t5.jpg)

首先复制，链表的结点，和不同链表一样，然后复制random指针。最后将两条链表分开。

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft3dhgukadj30kv0d1dg7.jpg)

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft3dhipp6gj30k40cit91.jpg)


## 面试题27 二叉搜索树和双向链表

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft3dzy2pm3j30l7058jrx.jpg)

采用中序遍历到结点4，pLastNodeList保存的是上一个结点，当pCurrent遍历到4时，pLastNodeList指向NULL，当pCurrent指向6时，pLastNodeList指向4，然后建立链表的双向链接，紧接着pCurrent指向8时，pLastNodeList指向6，然后pCurrent指向10，pLastNodeList指向8.总之就是每次都指向前面一个。

    void ConvertNode(TreeNode *pNode,TreeNode **pLastNodeinList)
    {
        if(pNode)
        {
            TreeNode *pCurrent=pNode;
            if(pCurrent->left!=NULL)
                ConvertNode(pCurrent->left,pLastNodeinList);
            pCurrent->left=*pLastNodeinList;
            if(*pLastNodeinList!=NULL)
                (*pLastNodeinList)->right=pCurrent;
            *pLastNodeinList=pCurrent;
            if(pCurrent->right!=NULL)
                ConvertNode(pCurrent->right,pLastNodeinList);
        }
    }

## 面试题28 字符串的全排列

回溯法

## 面试题29 数组中出现次数超过一半的数字

基于Partition的O(n)算法

超过一半，那么如果排好序，中间位置一定是出现次数最多的数。所以可以借鉴partition方法，寻找mid。不断地寻找index直到index=mid。

## 面试题30

和面试题29的方法一样，使index=k-1。这样左边每一个都比它小。因此前面k个即为最小的k个结点。

## 面试题31 连续子数组的最大和

输入一个数组，里面有正数也有负数，求中间连续序列的最大和。

动态规划：根据最大序列的特点，第一个序列肯定大于等于0。如果连续序列和小于0，那么说明前面一段可以放弃。形式化公式如下：

$$f(i)=data[i]; //f(i-1)<=0或者i=0$$
f(i)=f(i-1)+data[i]//i!=0或者f(i-1)>0

## 面试题32 整数中出现1的次数

常规方法：除以10取余

改进方法：？

## 面试题 33 丑数

对于数字中只包含2,3,5的数我们称之为丑数。

思路一：暴力求解。直接从头到尾遍历

思路二：借助数组保存前面的数。关系式如下面伪代码

    int nextUglyIndex=1;
    int *pMultiply2=pUglyNumbers;
    int *pMultiply3=pUglyNumbers;
    int *pMultiply5=pUglyNumbers;
    
    while(nextUglyIndex<index)
    {
        int min = Min(*pMultiply2*2,*pMultiply3*3,*pMultiply5*5);
        pUglyNumbers[nextUglyIndex]=min;
    
        while(*pMultiply2*2<=pUglyNumbers[nextUglyIndex])
            ++pMultiply2;
        while(*pMultiply3*3<=pUglyNumbers[nextUglyIndex])
            ++pMultiply3;
        while(*pMultiply5*5<=pUglyNumbers[nextUglyIndex])
            ++pMultiply5;
    
        ++nextUglyIndex;
    }

## 面试题35 数组中第一次出现的字符

采用hash表，统计每一个此处出现的次数然后，遍历hash，返回输出次数为1的表。

## 面试37 两个链表中的公共节点

首先一次求出两个链表的长度，然后计算出长度差d，让较长的链表的指针先移动n步。然后两个同时移动，两个链表第一次相等的结点即为公共节点的起始点。

## 面试题39 二叉树的深度

递归遍历，经典。需要熟记。

    int TreeDepth(TreeNode* pRoot)
    {
        int ldepth,rdepth;
        if(pRoot==NULL)
            return 0;
        ldepth = TreeDepth(pRoot->left)+1;
        rdepth = TreeDepth(pRoot->right)+1;
        return ldepth > rdepth?ldepth:rdepth;    
    }
    
    /*
    判断是否为平衡二叉树，平衡二叉树要求每一个结点的高度差不超过1
    */
    
    bool IsBalanced(TreeNode *pRoot)
    {
        if(pRoot==NULL)
            return true;
    
        int left=TreeDepth(pRoot->left);
        int right=TreeDepth(pRoot->right);
        int diff=left-right;
        if(diff>1||diff<-1)
            return false;
    
        return IsBalanced(pRoot->left)&&IsBalanced(pRoot->right);
    }

## 面试题40 数组中只出现一次的数字

一个整形数组中包含两个单独的数，其他的都是成对出现。

对于两个相同的数，其异或的值为0。

所以从头到尾一次异或数组中的数据，最后会得到一个不为0的数，即两个不成对的数的异或值，然后求出二进制异或值第一个不为0的位数，然后按照该位0~1分为两组，最后将两组分别进行异或操作最后得到的值即为两个单独的数。

## 面试题41 和为s的连续正序列

### 和为s的两个数字

输入一个递增排序的数组和数字s，在数组中查找两个数字，使得其和刚好等于s。

选择两个指针，分别指向数组头和尾，如果和大于s，尾指针向左移，否则头指针向右移。

### 和为s的序列

设置两个指针small和big。分别指向第一个位置和第二个位置。如果序列和小于s，big指针向后移，如果大于s，small指针向后移。最后得到序列。

## 面试题42 字符串翻转

## 面试题45 约瑟夫环

基础题，循环数组。

## 面试题46 1+2+3+...+n

利用构造函数

## 面试题47 不用加减乘除做加法

采用位运算

- 首先，异或计算求相加之后的值
- 然后与运算并且左移，计算进位
- 最后没有前面重复前面两步的操作，直到进位为0

伪代码


    int Add(int num1,int num2)
    {
        int sum,carry;
        do
        {
            sum=num1^num2;
            carry=(num1&num2)<<1;//求进位
            num1=sum;
            num2=carry;
    
        }while(num2!=0);
        return num1;
    }

## 面试题51 数组中重复的数字



## 面试题52 构建乘积数组

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft3ye9e801j30lz03jjti.jpg)

![](https://ws1.sinaimg.cn/large/005BVyzmgy1ft3yd60zbfj30ge0apdhk.jpg)

## 面试题53 正则匹配

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

- '.' 匹配任意单个字符
- '*' 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

根据分析其实可以得出p的第一个字符只能为'.'或者string字符，不能是'*'

正则匹配的遍历条件是两者同时匹配到终点。

​	如果两者同时到达结尾，则能匹配上，如果其中一个p已经遍历结束，但是s还没有遍历完则匹配不上

​	由于模式里面第一个不可能是'*'。判断的时候我们只需要从左往右判断就行。

​	如果p的第二字字符是*，则需要判断第一个字符是否能匹配上，如果能匹配上。则，模式有三种：

- 字符串向下移动，但是p不变。

- p移动两位，即*的中间匹配位数为0，s不动

- p不变，s向后移动。

  如果p的第二个字符不是*，判断就很简单，只需要一个一个往后面移动。

状态转移方程可以归结为
$$
f[i][j] = f[i-1][j] or f[i][j-2] or f[i-1][j-2]
$$

直接对照代码更容易理解



递归形式

~~~c
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

bool isMatch(char* str, char* pattern)
{
    if(str==NULL||pattern==NULL)
        return false;
    return matchCore(str,pattern);
}
~~~

动态规划

~~~python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i: int, j: int) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                # 如果p][j]='*'
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]
~~~



## 面试题56 链表中环的入口结点

思路：首先判断是否含有环，然后根据根据相关公式计算出入口节点。

具体做法首先设置两个指针，一个快指针和一个慢指针。快指针的速度为慢指针的速度的两倍，因此当他们能够相遇时，则存在入口节点，如果不能相遇，则不存在环。如果存在，则有下面的关系式。<font color=red>当找到相遇结点的时候，再将快速指针放到头结点，当两个同时再次相遇的时候，则会在环的入口处相遇</font>。[推导过程](https://blog.csdn.net/snow_7/article/details/52181049)

## 面试题58 二叉树下的一个结点

思路：
- 如果结点有右子树，那么下一个结点则是右子树中最左子树结点。
- 如果没有右子树，结点是它的父节点的左子节点，那么父节点则为下一个结点。
- 如果结点既没有右子树，同时它还是父节点的右子节点，那么一直沿着父节点向上遍历，直到找到一个是它父节点的左子节点。


## 面试题59 对称的二叉树

思路：对于对称二叉树而言，其原始二叉树和对称转换之后的二叉树的先序遍历顺序是一样的。因此采用先序遍历，先遍历左子树在遍历右子树得到的序列和先遍历左子树在遍历右子树得到的序列是一样的。

## 面试题60 按行打印二叉树

思路：按行打印二叉树需要利用队列先进先出的特点然后设置两个辅助变量，toBePrinted记录当前没有被打印的结点，nextLevel记录下一层需要打印的结点。首先将toBePrinted设置成1，nextLevel记录下来，然后判断父节点的叶子节点是否为空，不为空将其加入队列中，然后nextLevel+1。当该层需要打印的结点为0时，然后令toBePrinted=nextLevel，并且将nextLevel置为0。然后继续上述操作。

## 面试题61 按之字形顺序打印二叉树

之字形：和前面一题相似，按层输出二叉树，但是区别在于和正常的按层输出，奇数层是按照正序输出，偶数层是按照逆序输出。

思路：设计两个栈，奇数层的栈和偶数层的栈。首先将第一层节点压入奇数栈，然后将其弹出打印，并且按照左子节点右子节点的顺序将其子节点压入偶数栈，直到栈为空后换行。然后从偶数栈中依次弹出，并且将弹出的子节点按照右子节点和左子节点压入奇数栈。重复上面操作，直到所有站为空。

## 面试题62 序列化二叉树

常规题型，关键考察对于先序遍历二叉树以及递归的考虑。

## 面试题63 搜索二叉树的第k个结点

对于搜索二叉树，中序遍历是有序的，因此只需求出其中序遍历时的第k个数就行了。

## 面试题64 数据流中的中位数

对于一个有序数组，如果数据个数为奇数，那么中位数则为中间位置的数，如果为偶数，那么中位数则为中间两位的平均数。

这里采用大根堆和小根堆，对于中位数左边的一部分，将其保存在大根堆中，中位数右边的一部分保存到小根堆中。如果有奇数位数据，那么中位数在小根堆堆顶。如果有欧数位，那么中位数则为两个数据的平均值。为了使小根堆中每一个元素都大于大根堆。所以在添加数据的时候首先判断元素是否大于小根堆堆顶元素，如果大于则将其加入小根堆，如果小于则将其加入大根堆。并且需要判断小根堆和大根堆中数据个数之差是否小于2。如果等于2，需要将多的一个堆中的元素添加到少的堆的元素中。

    //利用STL实现小根堆和大根堆
    priority_queue<int, vector<int>, greater<int> > p;  // 小顶堆
    priority_queue<int, vector<int>, less<int> > q;     // 大顶堆

## 面试题65 滑动窗口的最大值

有一个数组，然后一个定长的滑动窗口按照固定的顺序滑动。要求求出每一次滑动过程中滑动窗口数据中的最大值。

思路一：直接采用暴力求解，将滑动窗口从头到尾滑动，然后求出每一次滑动时滑动窗口中包含的最大值。

思路二：采用双端队列求解问题。



## 面试题66 矩阵中的路径

和数据结构中[迷宫求解问题](https://blog.csdn.net/gzj_1101/article/details/46883051)相似。



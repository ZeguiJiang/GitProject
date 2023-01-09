/*
三元运算符

1:结构： （条件表达式），表达式1：表达式2

2.说明:
条件表达式结果为boolean类型
如果ture，就执行表达式1
如果false，执行表达式2

表达式1，2的要求是一致的




*/

class SanYuanTest{
    public static void main(String[] args) {
        //三元运算
        int m = 12;
        int n =10;
        int max = (m>n)? m:n;
        System.out.println(max);

        //三元运算符可以嵌套
        String maxstr = (m>n)? "mda":((m==n)?"m = n":"nda");
        System.out.println(maxstr);



        byte b = 2;
        System.out.println(b>m);


        int n1 = 13;
        int n2 = 30;
        int n3 = -100;
        int min = (n1<n2)? ((n1<n3)? n1:n3):n2;
        System.out.println(min);
        //能用三元的，一定可以if-else运算；
        //ifelse不一定能转化成三元、、

        //如果程序即可以三元，也可以ifelse， 我们优先选择三元运算

        //
        // practice
        int aa = 10, bb = 20, cc = 23;


        int max1 = (aa > bb)? aa:bb;
        int max2 = (cc > max1)? cc : max1;
        System.out.println(max2);

        //

        double d1  = 12.3, d2 = 12.9;
        double ans = (d1>10.0 && d2 < 20.0)? (d1+d2):(d1-d2);
        System.out.println(ans);







    }
} 
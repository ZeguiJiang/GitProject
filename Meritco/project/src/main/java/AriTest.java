/*
运算符之一，算术运算符
+-+-，前++


*/
public class AriTest{

        public static void main(String[] args) {
            int num1 = 12;
            int num2 = 5;
            int result = num1/num2;
            System.out.println(result);


            int result2 = num1/num2*num2;
            System.out.println(result2);

            double result3 = num1/num2;
            System.out.println(result3);

            double result4 = num1/(num2+0.0);

            //取余运算，结果符号·与被摩腹好相同



            // 前++， 先++后运算
            //后++， 先运算后++
            int a1 = 10;
            int b1 = ++a1;
            int a2 =10;
            int b2 = a2++;
            System.out.println("b1="+b1+"b1="+b2);



            short s1 = 10;
            s1 = (short)(s1+1);//因为加的int，所以需要强制转换
            System.out.println(s1);

            //自赠一不会改变本身的数据类型
            byte b = 127;
            b++;
            System.out.println(b);

            int a4 = 10;
            int b4 = --a4;
            System.out.println(b4);

            int test = 123;
            int test100 = test/100;
            int test10 = (test/10)%10;
            int test1 = test%10;
            System.out.println(test100+","+test10+","+test1);

            //必要的打字速度

        }

    }

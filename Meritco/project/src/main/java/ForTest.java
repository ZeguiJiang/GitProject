/*
循环结构
1初始化部分
2循环条件部分
3循环体部分
4迭代部分

for (1,2,4){
	3
}
*/
import java.util.Scanner;
class ForTest{
    public static void main(String[] args) {
        for(int i=1; i <= 10; i++){
            System.out.println(i);
        }

        //遍历100 内的偶数

        for(int i=1; i <=100; i++){
            if ((i %2) == 0){
                System.out.println(i);
            }
        }
    }
}

class PrimeTest{
    public static void main(String[] args) {
        Scanner scan  = new Scanner(System.in);
        System.out.println("输入一个数");
        int num = scan.nextInt();
        long start = System.currentTimeMillis();

        for (int i=2; i<=num; i++ ){
            Boolean is_prime = true;
            for (int j=2; j<i/2; j++){
                if( i%j == 0 ){
                    is_prime = false;
                    break;
                }
            }
            if (is_prime){
                System.out.println("the number    "+i+"   is prime");
            }
        }

        long end = System.currentTimeMillis();

        System.out.println("time spended is "+ (end-start));


    }
}
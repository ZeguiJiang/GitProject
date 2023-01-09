/* how to get variable from key board
 */
import java.util.Scanner;

public class ScannerTest{


    //具体实现步骤
    //1 导入包：import java.util.Scanner;、
    //2 scanner的实例化，Scanner scan = new Scanner(System.in);
    //3 调用scanner类的相关方法，来获取指定类型变量
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        System.out.println("your name?");
        String name = scan.next();
        System.out.println(" you name is " + name);

        System.out.println();
        System.out.println("what is your age?");
        int age = scan.nextInt();
        System.out.println("your age is " + age);


        System.out.println();
        System.out.println("what is your weight?");
        double weight = scan.nextDouble();
        System.out.println("your weight is " + weight);

        System.out.println();
        System.out.println("do you love me?(True/False)");
        boolean ans = scan.nextBoolean();
        System.out.println("your ans is" + ans);

        System.out.println();
        System.out.println("what is your score");
        int score = scan.nextInt();

		/*
		if(score == 100){
			System.out.println("you win a bmw");
		}else if(score <100 && score > 80){
			System.out.println("you win an iphone");
		}else if(score > 60 && score <=80){
			System.out.println("you pass");
		}else{
			System.out.println("you faild");
		}
		*/
        int num1 = 10, num3 = 11, num2 = 9;

        if (num1 > num2) {
            if (num2 > num3) {
                System.out.println(num1 + " " + num2 + " " + num3);
            } else if (num3 > num1) {
                System.out.println(num3 + " " + num2 + " " + num1);
            } else {
                System.out.println(num1 + " " + num3 + " " + num2);
            }
        } else {//num2 num1
            if (num3 > num2) {
                System.out.println(num1 + " " + num2 + " " + num3);
            } else if (num3 > num1) {
                System.out.println(num3 + " " + num2 + " " + num1);
            } else {
                System.out.println(num1 + " " + num3 + " " + num2);
            }
        }

    }
}


class ifelsetest{
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        System.out.println("tell me your score");
        int score = scan.nextInt();
        if (score == 100){
             System.out.println("奖励一辆保时捷");
        }else if (score > 80 && score <= 99){
            System.out.println("奖励一个ipad");
        }else if(score > 60 && score <= 80){
            System.out.println("还行");
        }else{
            System.out.println("奖励一顿暴打");
        }

        // 狗狗年龄测试
        System.out.println("你家狗多大了？");
        int dog_age = scan.nextInt();

        if (dog_age > 2){
            double hu_age = ( (dog_age - 2) * 4 ) + (2*10.5);
            System.out.println("拿它相当于人类"+hu_age+"岁");
        }else{
            double hu_age = (dog_age * 10.5);
            System.out.println("拿它相当于人类"+hu_age+"岁");
        }


    }

}


class ifelsetest1{
    /*
    假设你想开发一个玩彩票的游戏，程序随机地产生一个两位数的彩票，提示用户输入一个两位数，然后按照下面的规则判定用户是否能赢。

        1)如果用户输入的数匹配彩票的实际顺序，奖金10 000美元。
        2)如果用户输入的所有数字匹配彩票的所有数字，但顺序不一致，奖金 3 000美元。
        3)如果用户输入的一个数字仅满足顺序情况下匹配彩票的一个数字，奖金1 000美元。
        4)如果用户输入的一个数字仅满足非顺序情况下匹配彩票的一个数字，奖金500美元。
        5)如果用户输入的数字没有匹配任何一个数字，则彩票作废。
     */

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        // 随机生成数字
        int number = (int)(Math.random()*90 + 10);
        System.out.println("给一个你的数字，看看中没中奖, (0-99)");
        int mynum = scan.nextInt();
        int num_a = number/10, numb = number%10;
        int mynum_a = mynum/10, mynum_b = mynum%10;
        System.out.println("中奖数字是"+number );
        if(num_a == mynum_a && numb == mynum_b){
            System.out.println("奖金10000");
        }else if(num_a == mynum_b && numb == mynum_a){
            System.out.println("奖金3000");
        }else if (num_a == mynum_a || numb == mynum_b){
            System.out.println("奖金1000");
        }else if(num_a == mynum_b || numb == mynum_a){
            System.out.println("奖金500");
        }else{
            System.out.println("啥都没有");
        }

    }
}
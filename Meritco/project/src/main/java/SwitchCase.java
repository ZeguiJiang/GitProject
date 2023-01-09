/*
switch case：

结构：
switch(表达式){
	case 常量1：
	执行语句；
	break；

	case 常量2:
	执行语句
	break；

	default ：
	执行句子n

}



说明：
根据switch表达式的值，匹配各个case中的常量，一旦匹配成功，执行case中的语句，然后break退出，如果没有break，依次执行其他case结构中的语句，直到遇到break才结束。

break关键字在swicth case中使用，一旦遇到break，就跳出结构

switch结构中的表达式，只能是六种类型之一：
  byte，short，char，int，枚举类型（jdk5），string（jdk8）

 case 中只能声明常量

 default：相当于if else的 else结构

*/


import java.util.Scanner;

public class SwitchCase {

    public static void main(String[] args) {
        int num = 1;
        switch(num){
            case 0:
                System.out.println("0");
                break;
            case 1:
                System.out.println("1");
                break;
            default:
                System.out.println("other"+num);
                // no break needed
        }
    }
}


class SwitchCaseTest1{
    // 用switch case 来描述： 如果成绩大于60 就合格，否则不合格
    // 如果多个case 执行语句相同，可以考虑合并
    public static void main(String[] args) {
        int score = 6;

        switch(score/60){
            case 0:
                System.out.println("no pass");
                break;
            default:
                System.out.println("pass");
                // 根据月份，选择季节
        }
        int month = 9;
        switch(month){
            case 1:
            case 2:
            case 3:
                System.out.println("spring");
                break;
            case 4:
            case 5:
            case 6:
                System.out.println("summer");
                break;
            case 7:
            case 8:
            case 9:
                System.out.println("autume");
                break;
            default:
                System.out.println("winter");
        }
    }
}


class  SwitchCaseTest2{
    // 算算这个月过了多少天
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        System.out.println("which month");
        int month = scan.nextInt();

        System.out.println("which day");
        int day = scan.nextInt();
        int sumday = 0;
        switch(month){
            case 12:
                sumday = sumday + 30;
            case 11:
                sumday = sumday + 31;
            case 10:
                sumday = sumday + 30;
            case 9:
                sumday = sumday + 31;
            case 8:
                sumday = sumday + 31;
            case 7:
                sumday = sumday + 30;
            case 6:
                sumday = sumday + 31;
            case 5:
                sumday = sumday + 30;
            case 4:
                sumday = sumday + 31;
            case 3:
                sumday = sumday + 28;
            case 2:
                sumday = sumday+31;
            case 1:
                sumday = sumday + day;
                break;
        }
        System.out.println(sumday);

    }




}




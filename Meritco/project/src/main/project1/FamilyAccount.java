import java.util.Scanner;

public class FamilyAccount {
    //只做一个家庭account记账系统


    public static void main(String[] args) {
        String details = "收支\t\t账户金额\t\t收支金额\t\t说明\n";
        Scanner scan = new Scanner(System.in);
        int remain = 10000;
        details += "收入\t\t10000\t\t10000\t\t系统送的\n";
        Boolean exit = true;

        do {
            System.out.println("----------家庭收支系统----------");
            System.out.println("          1 收支明细");
            System.out.println("          2 登记收入");
            System.out.println("          3 登记支出");
            System.out.println("          4 退出程序");
            System.out.println("        ");
            System.out.println("          请输入执行的序号（1-4）：");
            int num = scan.nextInt();
            System.out.println();

            switch (num){
                case 1:
                    System.out.println("----------当前收支明细记录----------");
                    System.out.println(details);
                    System.out.println();
                    break;
                case 2:
                    System.out.println("2 登记收入");
                    System.out.println("请输入收入金额：");
                    int revenue = scan.nextInt();
                    System.out.println("请为此收入进行说明：");
                    String info = scan.next();
                    remain += revenue;
                    details += "收入\t\t"+remain+"\t\t"+revenue+"\t\t"+info+"\n";
                    //System.out.println(details);
                    break;
                case 3:
                    System.out.println("3 登记支出");
                    System.out.println("请输入支出金额：");
                    int spend = scan.nextInt();
                    System.out.println("请为此支出进行说明：");
                    String info_spend = scan.next();
                    remain -= spend;
                    details += "收入\t\t"+remain+"\t\t"+spend+"\t\t"+info_spend+"\n";
                    //System.out.println(details);
                    break;
                case 4:
                    System.out.println("你确定要退出吗（y/n）");
                    String check = scan.next();
                    if(check == "y") exit = false;
                    break;

            }



        } while (exit);

    }
}

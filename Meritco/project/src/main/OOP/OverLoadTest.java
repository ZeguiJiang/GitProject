
/*
 * 方法的重载（overload）  loading...
 *
 * 1.定义：在同一个类中，允许存在一个以上的同名方法，只要它们的参数个数或者参数类型不同即可。
 *
 *  "两同一不同":同一个类、相同方法名
 *            参数列表不同：参数个数不同，参数类型不同
 *
 * 2. 举例：
 *    Arrays类中重载的sort() / binarySearch()
 *
 * 3.判断是否是重载：
 *    跟方法的权限修饰符、返回值类型、形参变量名、方法体都没有关系！
 *
 * 4. 在通过对象调用方法时，如何确定某一个指定的方法：
 *      方法名 ---> 参数列表
 */



public class OverLoadTest {


    public static void main(String[] args) {
        OverLoadTest test = new OverLoadTest();

        int x = 10;
        int y = 20;
        int [] ids; //声明
        //1.1 静态初始化
        ids = new int[] {1001,1002,1003};
        int[] array = { 1, 3, 5 };

        int[] xlst = new int[2];
        xlst[0] = 3;
        xlst[1] = 5;
        int a2 = test.getSum(x, y);
        int a3 = test.getSum(xlst[0], xlst[1]);
        int a4 = test.getSum1(xlst);
        int aa = test.getSum(1,3);
        System.out.println(aa);
        System.out.println(a2);
        System.out.println(a3);
        System.out.println(a4);

        for(int i = 0; i < xlst.length; i++){
            System.out.println("xlsy"+ xlst[i]);
        }
    }
    //如下四个方法构成重载
    public int getSum(int i, int j){
        int ij = i + j;
        return ij;
    }

    public double getSum(double i, double j){
        double ij = i + j;
        return ij;
    }

    public double getSum(double i, int j){
        double ij = i + (double)(j);
        return ij;
    }

    public int getSum1(int[] lst){
        int tmp = 0;
        for(int i = 0; i < lst.length; i++){
            tmp+=lst[i];
            lst[i] = lst[i]+1;
            System.out.println("num"+lst[i]);
        }
        return tmp;
    }

}

/*
 数组的概念
    1 数组 多个同类型数据按照一定顺序排序的集合，并使用一个命名，并通过编号的方式统一管理
    2 数组的相关概念：
        数组名
        元素
        角标
        长度

    3 数组是有序排列
    4数组属于引用数据，数组的元素，既可以是基本数据类型，也可以是引用数据类型

 */


public class Array {
    public static void main(String[] args) {

        int num;
        num  = 10;
        int id = 1001;

        int [] ids; //声明
        //1.1 静态初始化
        ids = new int[] {1001,1002,1003};
        String[] names = new String[5]; //动态初始化长度为5的数组
        names[0] = "wangming";
        names[1] = "wanghe";
        names[2] = "zhangxueliang";
        names[4] = "jiangzegui";
        //names[5] = "ssss"; 会报错，超出范围， exception in thread

        //如何获取数组长度
        System.out.println(names.length);

        // 遍历数组
        for (int i=0; i<names.length; i++){
            System.out.println(names[i]);
        }






    }
}


class ArrayTest1{
    public static void main(String[] args) {
        //5 数组元素默认初始化值
        int[] arr = new int[4];
        for (int j=0; j < arr.length; j++){
            System.out.println(arr[j]);
        }


        char[] arr1 = new char[4];
        for (int j=0; j < arr1.length; j++){
            System.out.println(arr1[j]);
        }

        short[] arr2 = new short[4];
        for (int j=0; j < arr2.length; j++){
            System.out.println(arr2[j]);
        }

        double[] arr3 = new double[4];
        for (int j=0; j < arr3.length; j++){
            System.out.println(arr3[j]);
        }
    }
}

class ArrayDemo {
    public static void main(String[] args) {

            int[] arr = new int[]{8, 2, 1, 0, 3};
            int[] index = new int[]{2, 0, 3, 2, 4, 0, 1, 3, 2, 3, 3};
            String tel = "";
            for (int i = 0; i < index.length; i++) {
                tel += arr[index[i]];
            }
        System.out.println("联系方式：" + tel);
        }
}

class Array2D{
    public static void main(String[] args) {
        int[][] arr1 = new int[][]{{1,2,3},{4,5,6},{7,8,9}};
         //动态初始化
        String[][] arr2 = new String[3][2];


        //类型推断
        int[] arr23 = {12,3,5};
        int[] arrp[] = {{1,2},{3,4}};
        int arr22[][] = {{22},{222}};

        //元素调用
        System.out.println(arr1[1][1]);

        //元素遍历
        for(int i =0; i < arr1.length; i++){
            for(int j = 0; j < arr1[0].length; j++){
                System.out.println(arr1[i][j]);

            }
        }

    }
}
class YangHui{
    // 第一行有一个元素，第n行有n个元素

    //第一个和最后一个元素都是1
    //yanghui[i][j] = yanghui[i-1][j-1] + yanghui[i-1][j]
    public static void main(String[] args) {
        //初始化
        int[][] yanghui = new int[10][];
        //给元素赋值
        for(int i = 0; i<yanghui.length; i++){
            yanghui[i] = new int[i+1];

            yanghui[i][0] = yanghui[i][i] = 1;

            //内层for循环
            for( int j = 1; j < yanghui[i].length-1; j++){
                yanghui[i][j] = yanghui[i-1][j-1] + yanghui[i-1][j];

            }

        }

        //遍历数组
        for (int i =0; i<yanghui.length; i++){
            for (int j = 0; j < yanghui[i].length; j++){
                System.out.print(yanghui[i][j] + " ");
            }
            System.out.println( );
        }


    }
}

class ArrayCopy{
    //有效复制
    public static void main(String[] args) {
        int[] array1 , array2;
        array1 = new int[] {1,2,3,4};
        array2 = new int[array1.length];

        //数组复制
        for (int i =0; i<array1.length; i++){
            array2[i] = array1[i];
        }


        //s数组反转
        //int array_inverse;

        for (int i =0; (i<array1.length / 2); i++){
            int tmp = array1[i];
            array1[i] = array1[array1.length - i - 1 ];
            array1[array1.length - i - 1 ] = tmp;

        }

    }
}
public class VariableTest {
    public static void main(String[] args) {


        /* 凡事可以起名的地方，都叫标识符
         */
        //变量的定义
        int mynumber =  1000;
        System.out.println(mynumber);


        //变量的声明
        int myNum;
        //变量的赋值
        myNum = 100;
        System.out.println(myNum);

        byte b1 = -120;
        long l1 = 324798l;
        System.out.println(b1);
        System.out.println(l1);

		/*浮点型：
		float 可以精确7位 / 四个字节 / 表示数值范围比long还大
		double可以精确14位 / 八个字节 /

		*/
        double d1 = 103.33;
        System.out.println(d1 + 12.2);
        //定义float时，要以f或者F结尾
        float f1 = 12.3f;
        System.out.println(f1);

		/*字符型
		char 1字符 = 2字节 ， 通常使用一对单引号
		*/
        char c1 = 'a';
        System.out.println(c1);

        //换行符，制表符
        char ch = '\n';
        char cb = '\t';

        //unicode 字符的编码集
        char cc = '\u0043';
        System.out.println("ssss"+cb+"dddd"+cc);

  		/*布尔型 boolean
  		只有true false
  		通常条件判断，许循环语句
  		*/
        boolean isMarried = true;
        if(isMarried){
            System.out.println("滚吧,\"no coming\"");
        }else{
            System.out.println("with me");
        }
    }


}

class VariableTest1 {
    public static void main(String[] args) {
        //自动类型提升：
        //当容量小的数据类型变量和容量大的数据类型变量做运算的时候，结果自动提升为容量大的数据类型
        // byte，char，short --int--long--float--double 从小到大
        //byte，char，shor做运算，结果是int

        byte b1 = 2;
        int i1 = 12;
        int b2 = i1+b1;
        System.out.println(b2);
        float b3 = i1+100;
        System.out.println(b3);
        //**************************

        char c1 = 'a';
        int i2 = 10;
        int i4 = c1+i2;//最小写int
        System.out.println(i4);
    }
}

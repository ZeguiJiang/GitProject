/*



 */



public class Hello {
    public static void main(String[] args) {
        // 单行注释
        // 解释说明，增强粘性
        // 方便自己，方便别人
        System.out.println("Hello");

       Person1 jack =  new Person1();
       jack.studentName = "jack";
       jack.studentAge = 10;
       jack.address = "hainan";
       System.out.println("姓名：" + jack.studentName);
        System.out.println("年龄：" + jack.studentAge);
        System.out.println("地址：" + jack.address);
    }
}

class Person1{
    int studentAge;
    String studentName;
    String address;

}
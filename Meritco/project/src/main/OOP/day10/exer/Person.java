package day10.exer;

public class Person {
    private int age;
    public void setAge(int a){
        if (a < 0 || a > 130){
            // throw new runtime Exception("传入数据非法"）
            System.out.println("c传入数据非法");
            return;

        }

        age = a;
    }

    public int getAge(){
        return age;
    }
}

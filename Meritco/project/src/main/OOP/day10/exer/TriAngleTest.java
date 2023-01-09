package day10.exer;

public class TriAngleTest {

  public static void main(String[] args) {
    TriAngle t1 = new TriAngle();
    t1.setBase(2.0);
    t1.setHeight(3.0);

    // t1.base //私有化不可以调用
    System.out.println("base:"+ t1.getBase() + "  , height:" + t1.getHeight());
  }
}

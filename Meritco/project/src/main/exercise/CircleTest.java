/*

利用面向对象，设计计算圆的面积



 */


public class  CircleTest{
    public static void main(String[] args) {
        Circle cc = new Circle();
        cc.radius = 5.5;
        double aa = cc.Area();
        System.out.println(aa);

    }

}


class Circle {
    //属性
    double radius;
    //方法

    public double Area(){
        double area = 3.14 * radius * radius;
        return area;
    }

}

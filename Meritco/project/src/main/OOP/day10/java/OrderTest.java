package day10.java;


import day10.java.Order;

public class OrderTest {
    public static void main(String[] args) {
        Order order = new Order();

        order.orderDefault = 1;
        order.orderPublic = 2;

        // 除了order类之后，私有结构就不可以调用了
        //order.orderPrivate = 3;

        order.methodDefault();
        order.methodPublic();
        //出了order类之后，私有结构就不可以调用了
        //order.methodPrivate();


    }
}

/*logic test
逻辑运算符操作的是boolean类型 

*/

class LogicTest{
    public static void main(String[] args) {
        //区分逻辑语，& and &&
        //相同点：运算结果相同； 当左边符号是true时，二者都会执行符号右边的运算

        //不同点，当左边符号是false时，单&会执行符号右边的运算，双则不会


        boolean b1 = true;
        int a1  = 10;
        if(b1&(a1++>0)){
            System.out.println("beijing");
        }else{
            System.out.println("nanjing");
        }

        System.out.println(a1);


        boolean b2 = true;  
        int a2  = 10;
        if(b2&&(a2++>0)){
            System.out.println("beijing");
        }else{
            System.out.println("nanjing");
        }
        System.out.println(a2);





        //区分逻辑语，｜ and ｜｜
        //相同点：运算结果相同； 当左边符号是false时，二者都会执行符号右边的运算

        //不同点，当左边符号是ture时，单|会执行符号右边的运算，双则不会

        boolean b3 = false;
        int a3  = 10;
        if(b3|(a3++>0)){
            System.out.println("beijing");
        }else{
            System.out.println("nanjing");
        }

        System.out.println(a3);


        boolean b4 = false;
        int a4  = 10;
        if(b4||(a4++>0)){
            System.out.println("beijing");
        }else{
            System.out.println("nanjing");
        }

        System.out.println(a4);

    }
}
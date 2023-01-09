import sun.lwawt.macosx.CSystemTray;

public class StudentTest {
    public static void main(String[] args) {
        Student[] students = new Student[20];

        // 生成20个学生
        for(int i = 0; i<20; i++){
            students[i] = new Student();
            students[i].age = (int) (Math.random() * (6) + 6 );
            students[i].score =(int) (Math.random() * 100 + 1);
            students[i].id =(int) (i + 1);

        }

//        for (int i=0; i<students.length; i++){
//            System.out.print("  ｜ 学生年龄" + students[i].age);
//            System.out.print("  ｜ 学生成绩" + students[i].score);
//            System.out.print("  ｜ 学生id" + students[i].id);
//            System.out.println();
//        }

        StudentTest test = new StudentTest();
        test.print_stu_info(students);



    }

    public void print_stu_info(Student[] students){
        for (int i=0; i<students.length; i++){
            System.out.print("  ｜ 学生年龄" + students[i].age);
            System.out.print("  ｜ 学生成绩" + students[i].score);
            System.out.print("  ｜ 学生id" + students[i].id);
            System.out.println();
        }
    }
}


class Student{
    //年龄。 6-12岁
    int age;
    //成绩， 1-100
    int score;
    //学生id
    int id;


}

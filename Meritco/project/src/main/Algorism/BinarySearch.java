
import java.util.Scanner;
public class BinarySearch {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int[] arr = new int[] {1,2,3,4,5,6,7,8,10};
        System.out.println("please enter the number you want to search");
        int num = scan.nextInt();
        int start = 0, end = arr.length;
        boolean flag = false;

        // 1,2,3,4,5,6,7,8,9
        // s m    e m       e

        while(start < end){
            int mid = (start + end) / 2;
            if (arr[mid] > num){
                end = mid-1;
            }else if(arr[mid] < num){
                start = mid+1;
            }else{
                System.out.println("the index is "+mid);
                flag = true;
                break;
            }
        }

        if(! flag){
            System.out.println("not found");
        }
    }

}
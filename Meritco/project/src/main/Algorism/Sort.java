


public class Sort {
    //bubble sort
    public static void main(String[] args) {
        int[] arr = new int[] {91,2,22,11,5,32,77,8,10};

        for(int i=0; i < arr.length ; i++){
            //每一轮
            for (int j = 0; j < arr.length - i-1;j++ ){
                //这一轮进行bubble 排序
                //最大的换到最后边
                if(arr[j]> arr[j+1]) {
                    int tmp = arr[j + 1];
                    arr[j + 1] = arr[j];
                    arr[j] = tmp;

                }
            }
        }
        for(int i=0; i < arr.length ; i++){
            System.out.print(arr[i]+"\t");
        }

    }
}

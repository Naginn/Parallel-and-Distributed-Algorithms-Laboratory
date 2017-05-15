package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Scanner;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RunMe {

    public static void main(String args[]) throws InterruptedException, ExecutionException, IOException {
        
        int threadCount = 4;
        int rows = 0;
        
        ArrayList<ArrayList<Integer>> a = new ArrayList<ArrayList<Integer>>();
        ArrayList<ArrayList<Integer>> b = new ArrayList<ArrayList<Integer>>();
        
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        Collection<Task> collection = new ArrayList<Task>( );
        
        Scanner input = null;
        input = new Scanner(new File("file1.txt"));     
            
        while(input.hasNextLine()) {
            rows++;
            Scanner colReader = new Scanner(input.nextLine());
            ArrayList col = new ArrayList();
            while(colReader.hasNextInt()) {
                col.add(colReader.nextInt());       
            }
            a.add(col);
        }
        
        input = new Scanner(new File("file2.txt"));    

        while(input.hasNextLine()) {
            Scanner colReader = new Scanner(input.nextLine());
            ArrayList col = new ArrayList();
            while(colReader.hasNextInt()) {
                col.add(colReader.nextInt());       
            }
            b.add(col);
        }
        
        ArrayList<ArrayList<Integer>> c = new ArrayList<ArrayList<Integer>>();

        for(int i = 0; i < rows; i++){
            ArrayList col1 = new ArrayList();
            for(int j = 0; j < a.get(0).size(); j++){
                col1.add(0);
            }
             c.add(col1);
        }
        
        for(int i = 0; i < rows ; i++){
            Task task = new Task(i, a, b, c);
            collection.add(task);
        }

        executor.invokeAll(collection);
        executor.shutdown();

        Writer wr  = new FileWriter("file3.txt");

        for(ArrayList item: c){
            for(Object item2: item){
                wr.write(item2.toString() + " ");
            }
            wr.write(" ");
        }

        System.out.print("Terminat!");
    }
}

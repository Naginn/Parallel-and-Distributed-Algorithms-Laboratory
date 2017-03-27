package main;

import java.util.ArrayList;
import java.util.concurrent.Callable;

public class Task implements Callable<Object> {

    int x ;
    ArrayList<ArrayList<Integer>> a;
    ArrayList<ArrayList<Integer>> b;
    ArrayList<ArrayList<Integer>> c;
    
    
    public Task(int x, ArrayList<ArrayList<Integer>> a, ArrayList<ArrayList<Integer>> b, ArrayList<ArrayList<Integer>> c){
        this.x = x;
        this.a = a;
        this.b = b;
        this.c = c;
    }

    @Override
    public Object call() {
        // TODO Auto-generated method stub
        for(int i = 0; i < a.get(0).size(); i++){
            c.get(x).set(i, a.get(x).get(i) + b.get(x).get(i));
        }
        return null;
    }
}
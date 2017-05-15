package Semaphore;

import java.util.LinkedList;
import java.util.concurrent.Semaphore;

public class ConsumerProducer {

    static Object LOCK = new Object();
    static LinkedList list = new LinkedList();
    static Semaphore sem = new Semaphore(0);
    static Semaphore mutex = new Semaphore(1);

    static class Producer extends Thread {
        public void run() {
            try {
                int N = 0;
                
                while (true) {
                    mutex.acquire();
                    list.add(new Integer(N++));
                    mutex.release();
                    sem.release(1);
                    Thread.sleep(1000);
                }
            } catch (Exception x) {
                x.printStackTrace();
            }
        }
    }
    static class Consumer extends Thread {
        Integer nr;
        public Consumer(Integer nr) {
            this.nr = nr;
        }
        public void run() {
            try {

                while (true) {
                    sem.acquire(1);
                    mutex.acquire();
                    System.out.println("Consumer \""+nr+"\" cosumed "+list.removeFirst());
                    mutex.release();
                }
            } catch (Exception x) {
                x.printStackTrace();
            }
        }
    }

    public static void main(String [] args) {
        new Producer().start();
        new Consumer(1).start();
        new Consumer(2).start();
    }
}
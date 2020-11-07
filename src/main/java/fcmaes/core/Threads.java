/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

/**
 * Threading utility class. 
 */

public class Threads {

    Thread[] threads;

    public Threads(Runnable runnable, int numThreads) {
        int n = numThreads;
        threads = new Thread[n];
        for (int i = 0; i < n; i++)
            threads[i] = new Thread(runnable);
    }
    
    public Threads(Runnable runnable) {
        int n = Runtime.getRuntime().availableProcessors();
        threads = new Thread[n];
        for (int i = 0; i < n; i++)
            threads[i] = new Thread(runnable);
    }

    public Threads(Runnable[] runnables) {
        int n = runnables.length;
        threads = new Thread[n];
        for (int i = 0; i < n; i++)
            threads[i] = new Thread(runnables[i]);
    }

    public void start() {
        for (Thread thread : threads)
            thread.start();
    }

    public void join() {
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
            }
        }
    }

}

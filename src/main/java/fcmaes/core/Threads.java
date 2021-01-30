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

    public Threads(Runnable runnable, int numWorkers) {
        int n = numWorkers;
        threads = new Thread[n];
        for (int i = 0; i < n; i++)
            threads[i] = new Thread(runnable);
    }

    public static int numWorkers() {
//      return Math.min(16, Runtime.getRuntime().availableProcessors());
        return Math.min(32, Runtime.getRuntime().availableProcessors());
    }

    public Threads(Runnable runnable) {
        int n = numWorkers();
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

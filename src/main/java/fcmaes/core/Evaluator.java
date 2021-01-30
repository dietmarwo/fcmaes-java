package fcmaes.core;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Parallel evaluation of the population of an optimization algorithm
 * in parallel. Threads are kept alive during the whole optimization 
 * process to avoid costly repeated thread recreation. 
 * Uses BlockingQueue to synchronize the parallel evaluations. 
 * 
 * Note that this shouldn't be used for parallel optimization retry. In general 
 * parallel optimization retry scales better than parallel function evaluation. 
 */

public class Evaluator {
    
    Fitness fit;
    int workers;
    int popsize;
    BlockingQueue<Integer> task;
    BlockingQueue<Integer> finished;
    double[][] xs;
    double[] ys;
    Runner runner = null;
    Threads threads;
    boolean stop = false;
    
    public Evaluator(Fitness fit, int popsize, int workers) {
        this.fit = fit;
        if (workers <= 0 || workers > Threads.numWorkers()) // set default and limit
            workers = Threads.numWorkers();
        this.workers = workers;
        this.popsize = popsize;
        this.task = new ArrayBlockingQueue<Integer>(popsize);
        this.finished = new ArrayBlockingQueue<Integer>(popsize);
        this.runner = new Runner();
        this.threads = new Threads(this.runner, workers);
        this.threads.start();
    }

    public void destroy() {
        stop = true;
        try {
            for (int i = 0; i < workers; i++)
                task.put(i);
            threads.join();       
        } catch (InterruptedException e) {
        }  
     }

    protected void finalize() throws Throwable {
        destroy();
    }
    
    public synchronized double[] eval(double[][] xs) {
       int num = xs.length;
       try {
           this.xs = xs;
           ys = new double[num];
           for (int i = 0; i < num; i++)
               task.put(i);
           for (int i = 0; i < num; i++)
               finished.take();
           return ys;
       } catch (InterruptedException e) {
           return Utils.array(num, Double.MAX_VALUE);
       }      
    }

    private class Runner implements Runnable {
        
        @Override
        public void run() {
            try {
                while(!stop) {
                    int i = task.take();
                    if (!stop) {
                        try {
                            ys[i] = fit.value(xs[i]);
                        } catch (Exception ex) {
                            ys[i] = Double.MAX_VALUE;
                        }
                        finished.put(i);
                    }
                }
            } catch (InterruptedException ex) {
                throw new RuntimeException("unexpected interrupt");
            }
        }
    }  

}

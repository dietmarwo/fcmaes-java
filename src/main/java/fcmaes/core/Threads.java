/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

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

    public static void execute(Runnable[] runnables) {
    	int num = numWorkers();
		for (int i = 0; i < runnables.length; i += num) {	    		
	    	Threads threads = new Threads(
	    			Arrays.copyOfRange(runnables, i, Math.min(runnables.length, i+num)));
	    	threads.start();
	    	threads.join();
		}
    }
    
    public static void executeQueue(Runnable[] runnables) {
    	Runner runner = new Runner(runnables);
    	runner.join();
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
    
    private static class Runner {
    	BlockingQueue<Integer> requests;
    	Runnable[] runnables;
    	List<Job> jobs;
    	boolean stop = false;
    	AtomicInteger next;

    	public Runner(Runnable[] runnables) {
    		this.runnables = runnables;
    		int n = runnables.length;
    		int workers = 32;//16;//Math.min(n, Threads.numWorkers());
       	    this.requests = new ArrayBlockingQueue<Integer>(workers);
    		for (int i = 0; i < workers; i++)
    			requests.add(i);
    		next = new AtomicInteger(workers);
    		jobs = new ArrayList<Job>(workers);
    		// create and start jobs;
    		for (int i = 0; i < workers; i++)
    			jobs.add(new Job(this));
    	}
    	
       	void run() {
    		try {
    			while (!stop) {
    				int i = requests.take();
    				if (stop) {
    					requests.add(i);
    				} else {
    					try {
    						if (i < runnables.length)
    							runnables[i].run();
    						else 
    							stop = true;
    						int nextI = next.getAndIncrement();
    						requests.add(nextI);
    					} catch (Exception e) {
    						System.err.println(e.getMessage());
    					}
    				}
    			}
    		} catch (InterruptedException e) {
    		}
    	}

    	public void join() {
    		for (Job job : jobs) {
    			job.join();
    		}
    	}
    	
    	public void finalize() {
    		stop = true;
    		join();
    	}

    }

	private static class Job {

		Thread thread;

		Job(Runner runner) {
			thread = new Thread() {
				public void run() {
					runner.run();
				}
			};
			thread.start();
		}

		void join() {
			try {
				thread.join();
			} catch (InterruptedException e) {
			}
		}

	};

}

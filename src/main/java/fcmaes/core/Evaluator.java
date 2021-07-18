package fcmaes.core;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Parallel evaluation of the population of an optimization algorithm. 
 * Threads are kept alive during the whole optimization process to
 * avoid costly repeated thread recreation. Uses BlockingQueue to synchronize
 * the parallel evaluations.
 * 
 * Note that this shouldn't be used for parallel optimization retry. In general
 * parallel optimization retry scales better than parallel function evaluation.
 */

public class Evaluator {

	static class VecId {
		int id;
		double[] v;

		VecId(double[] v_, int id_) {
			id = id_;
			v = v_;
		}

		VecId() {
		}
	}

	private static class Job {

		Thread thread;

		Job(int id, Evaluator exec, boolean isMO) {
			thread = new Thread() {
				public void run() {
					if (isMO)
						exec.executeMO(id);
					else
						exec.execute(id);
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

	Fitness fit;
	int workers;
	BlockingQueue<VecId> requests;
	BlockingQueue<VecId> evaled;
	boolean stop = false;
	List<Job> jobs;

	public Evaluator(Fitness fit, int popsize, int workers) {
		this.fit = fit;
		int maxWorkers = Math.min(popsize, Threads.numWorkers());
		if (workers <= 0 || workers > maxWorkers) // set default and limit
			workers = Threads.numWorkers();
		this.workers = workers;
		this.requests = new ArrayBlockingQueue<VecId>(2 * workers);
		this.evaled = new ArrayBlockingQueue<VecId>(2 * workers);
		jobs = new ArrayList<Job>(2 * workers);
		for (int i = 0; i < 2 * workers; i++)
			jobs.add(new Job(i, this, fit instanceof FitnessMO));
	}

    void evaluate(double[] x, int id) {
    	requests.add(new VecId(x, id));
    }

    VecId result() {
    	try {
			return evaled.take();
		} catch (InterruptedException e) {
			e.printStackTrace();
			return null;
		}
    }

	public void join() {
		stop = true;
		// to release all locks
		for (Job job : jobs) {
			requests.add(new VecId());
		}
		for (Job job : jobs) {
			job.join();
		}
	}

	public void finalize() {
		join();
	}

	void execute(int threadId) {
		try {
			while (!stop) {
				VecId vid = requests.take();
				if (!stop) {
					try {
						double y = fit.value(vid.v);
						vid.v = new double[] { y };
					} catch (Exception e) {
						System.err.println(e.getMessage());
						vid.v = new double[] { Double.MAX_VALUE };
					}
					evaled.offer(vid, 10000000, TimeUnit.DAYS);
				}
			}
		} catch (InterruptedException e) {
		}
	}
	
	void executeMO(int threadId) {
		FitnessMO fitmo = (FitnessMO)fit;
		try {
			while (!stop) {
				VecId vid = requests.take();
				if (!stop) {
					try {
						double[] y = fitmo.movalue(vid.v);
						vid.v = y;
					} catch (Exception e) {
						System.err.println(e.getMessage());
						vid.v = Utils.array(fitmo._nobj, Double.MAX_VALUE);
					}
					evaled.add(vid);
				}
			}
		} catch (InterruptedException e) {
		}
	}

}

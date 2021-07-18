/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import org.apache.commons.lang3.mutable.MutableInt;

import fcmaes.core.Optimizers.Result;

/**
 * Java wrapper for the C++ CMA-ES implementation to support an ask/tell 
 * interface and an alternative parallel function evaluation implementation.
 * 
 * Note that new CMA().minimize is faster because it involves less JNI overhead.
 * 
 * Note that minimize_parallel shouldn't be used for parallel optimization retry. 
 * In general parallel optimization retry scales better than parallel function evaluation. 
 */

public class Cmaes {

    private long nativeCmaes;

    public static Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
            int maxIter, int maxEvals, double stopValue, int popsize, int mu, double accuracy, long seed, int runid,
            boolean normalize, int update_gap, int workers) {
    	if (workers <= 1) {
    		Jni.optimizeACMA(fit, lower, upper, sigma, guess, 1000000, maxEvals, stopValue, popsize,
    				popsize / 2, accuracy, seed, runid, normalize, update_gap, 1);
    		return new Result(fit, fit._evals);
    	} else {
    		return minimize_parallel(fit, lower, upper,sigma, guess,
    	            maxEvals, stopValue, popsize, mu, accuracy, seed, runid,
    	            normalize, update_gap, workers);
    	}
    }

    /**
     * Create a CMA-ES object for ask/tell.
     * 
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     */

    public Cmaes(Fitness fit, double[] lower, double[] upper) {
        int popsize = 31;
        double[] guess = Utils.rnd(lower, upper);
        double[] sigma = Utils.array(lower.length, Utils.rnd(0.05, 0.1));
        nativeCmaes = Jni.initCmaes(fit, lower, upper, sigma, guess, popsize, popsize / 2, 1.0, Utils.rnd().nextLong(), 0, true,
                -1);
    }

    /**
     * Create a CMA-ES object for ask/tell.
     * 
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     * @param sigma    Individual input sigma.
     * @param guess    Starting point.
     * @param popsize  Population size used for offspring.
     */

    public Cmaes(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess, int popsize) {
        if (popsize <= 0)
            popsize = 31;
        if (guess == null)
            guess = Utils.rnd(lower, upper);
        if (sigma == null)
            sigma = Utils.array(lower.length, Utils.rnd(0.05, 0.1));
        nativeCmaes = Jni.initCmaes(fit, lower, upper, sigma, guess, popsize, popsize / 2, 1.0, Utils.rnd().nextLong(), 0, true,
                -1);
    }

    /**
     * Create a CMA-ES object for ask/tell.
     * 
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     * @param sigma    Individual input sigma.
     * @param guess    Starting point.
     * @param popsize  Population size used for offspring.
     * @param mu  	   Number of parents/points for recombination.
     * @param accuracy default is 1.0.
     * @param seed     Random seed.
     * @param runid    id for debugging/logging.
     * @param normalize if > 0 geno transformation maps arguments to interval [-1,1].
     * @param update_gap number of iterations without distribution update, use 0 for default.
     */

    public Cmaes(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess, int popsize, int mu, double accuracy,
            long seed, int runid, boolean normalize, int update_gap) {
        nativeCmaes = Jni.initCmaes(fit, lower, upper, sigma, guess, popsize, mu, accuracy, seed, runid, normalize,
                update_gap);
    }
    
    /**
     * Ask for argument vector.
     * 
     * @return  		Argument vector for evaluation.
     */

    public double[] ask() {
        return Jni.askCmaes(nativeCmaes);
    }

    /**
     * Tell evaluated argument vector.
     * 
     * @param x      	Argument vector.     
     * @param y      	Function value.     
     * @return  		Status of optimization, stop if > 0.
     */

    public int tell(double[] x, double y) {
        return Jni.tellCmaes(nativeCmaes, x, y);
    }

    /**
     * Destroy corresponding native C++ object 
     * to avoid a memory leak
     */

    public void destroy() {
        Jni.destroyCmaes(nativeCmaes);
    }

    protected void finalize() throws Throwable {
        destroy();
    }
    
    public double[] population() {
        return Jni.populationCmaes(nativeCmaes);
    }

    /**
     * Minimize evaluating the fitness function in parallel.
     * 
     * @param fit      fitness function.     
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     * @param sigma    Individual input sigma.
     * @param guess    Starting point.
     * @param maxEvals Maximum number of evaluations.
     * @param stopValue target function value.
     * @param popsize  Population size used for offspring.
     * @param mu  	   Number of parents/points for recombination.
     * @param accuracy default is 1.0.
     * @param seed     Random seed.
     * @param runid    id for debugging/logging.
     * @param normalize if > 0 geno transformation maps arguments to interval [-1,1].
     * @param update_gap number of iterations without distribution update, use 0 for default.
     * @param workers  number of parallel threads for function evaluations.
     * @return Result  Minimized function value / optimized point.
     */

    public static Result minimize_parallel(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
            int maxEvals, double stopValue, int popsize, int mu, double accuracy, long seed, int runid,
            boolean normalize, int update_gap, int workers) {
        int dim = fit._dim;
        if (guess == null)
            guess = Utils.rnd(lower, upper);
        if (sigma == null)
            sigma = Utils.array(dim, 0.3);
        Cmaes opt = new Cmaes(fit, lower, upper, sigma, guess, popsize, mu, accuracy, seed, runid, normalize, update_gap);

        int workerLimit = Math.min(popsize, Threads.numWorkers());
		if (workers <= 0 || workers > workerLimit) // set default and limit
			workers = workerLimit;
		Evaluator evaluator = new Evaluator(fit, popsize, workers);
		int stop = 0;
		fit._evals = 0;
		double[][] evals_x = new double[popsize][];
		// fill eval queue with initial population
		for (int i = 0; i < workers; i++) {
			double[] x = opt.ask();
			evaluator.evaluate(x, i);
			evals_x[i] = x;
		}
		while (fit._evals < maxEvals && stop == 0) {
			Evaluator.VecId vid = evaluator.result();
			double y = vid.v[0]; // single objective
			int p = vid.id;
			double[] x = evals_x[p];
			opt.tell(x, y);
			x = opt.ask();
			evaluator.evaluate(x, p);
			evals_x[p] = x;
		}
		evaluator.join();
		return new Result(fit, fit._evals);
	}
}

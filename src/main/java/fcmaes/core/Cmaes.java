/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

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

    public Cmaes(double[] lower, double[] upper, double[] sigma, double[] guess, int popsize, int mu, double accuracy,
            long seed, int runid, int normalize, int update_gap) {
        nativeCmaes = Jni.initCmaes(lower, upper, sigma, guess, popsize, mu, accuracy, seed, runid, normalize,
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
            int normalize, int update_gap, int workers) {
        int dim = fit._dim;
        Cmaes opt = new Cmaes(lower, upper, sigma, guess, popsize, mu, accuracy, seed, runid, normalize, update_gap);
        if (workers <= 0 || workers > Threads.numWorkers()) // set default and limit
            workers = Threads.numWorkers();
        Evaluator evaluator = new Evaluator(fit, popsize, workers);
        int evals = 0;
        int stop = 0;        
        for (; evals < maxEvals && stop == 0; evals += workers) {
            double[][] xs = new double[workers][dim];
            for (int p = 0; p < workers; p++)
                xs[p] = opt.ask();
            double[] ys = evaluator.eval(xs);
            for (int p = 0; p < workers && stop == 0; p++)
                stop = opt.tell(xs[p], ys[p]);
            if (fit._bestY < stopValue)
            	break;
        }
        evaluator.destroy();
        return new Result(fit, fit._evals);
    }
}

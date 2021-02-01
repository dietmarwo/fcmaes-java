/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

/**
 * Java wrapper for the C++ differential evolution implementation to support an ask/tell 
 * interface and an alternative parallel function evaluation implementation.
 * 
 * Note that new DE().minimize is faster because it involves less JNI overhead.
 * 
 * Note that minimize_parallel shouldn't be used for parallel optimization retry. 
 * In general parallel optimization retry scales better than parallel function evaluation. 
 */

import java.util.Arrays;

import org.apache.commons.lang3.mutable.MutableInt;

import fcmaes.core.Optimizers.Result;

public class De {
	
    private long nativeDe;

    /**
     * Create a DE object for ask/tell.
     * 
     * @param lower    	lower point limit.
     * @param upper    	upper point limit.
     * @param guess    	Starting point.
     * @param popsize  	Population size used for offspring.
     * @param keep  	Changes the reinitialization probability of individuals based on their age. 
     * 					Higher value means lower probability of reinitialization.
     * @param F  		The mutation constant. In the literature this is also known as differential weight, 
        		 		being denoted by F. Should be in the range [0, 2].
     * @param CR  		The recombination constant. Should be in the range [0, 1].
     * @param seed  	Random seed.
     * @param runid    	id for debugging/logging.
     */

    public De(double[] lower, double[] upper, double[] guess, int popsize, double keep, double F, double CR, long seed,
            int runid) {
        nativeDe = Jni.initDE(lower, upper, guess, popsize, keep, F, CR, seed, runid);
    }
    
    /**
     * Ask for argument vector.
     * 
     * @param pos      	Set to position of argument.     
     * @return  		Argument vector for evaluation.
     */

    public double[] ask(MutableInt pos) {
        double[] asked = Jni.askDE(nativeDe);
        int dim = asked.length - 1;
        pos.setValue((int) asked[dim]);
        return Arrays.copyOfRange(asked, 0, dim);
    }

    /**
     * Tell evaluated argument vector.
     * 
     * @param x      	Argument vector.     
     * @param y      	Function value.     
     * @param p      	Position of argument.     
     * @return  		Status of optimization, stop if > 0.
     */

    public int tell(double[] x, double y, int p) {
        return Jni.tellDE(nativeDe, x, y, p);
    }

    /**
     * Destroy corresponding native C++ object 
     * to avoid a memory leak
     */

    public void destroy() {
        Jni.destroyDE(nativeDe);
    }

    protected void finalize() throws Throwable {
        destroy();
    }

    /**
     * Minimize evaluating the fitness function in parallel.
     * 
     * @param fit      	fitness function.     
     * @param lower    	lower point limit.
     * @param upper    	upper point limit.
     * @param guess    	Starting point.
     * @param maxEvals 	Maximum number of evaluations.
     * @param stopfitness target function value.
     * @param popsize  	Population size used for offspring.
     * @param keep  	Changes the reinitialization probability of individuals based on their age. 
     * 					Higher value means lower probability of reinitialization.
     * @param F  		The mutation constant. In the literature this is also known as differential weight, 
        		 		being denoted by F. Should be in the range [0, 2].
     * @param CR  		The recombination constant. Should be in the range [0, 1].
     * @param seed  	Random seed.
     * @param runid    	id for debugging/logging.
     * @param workers 	number of parallel threads for function evaluations.
     * @return Result 	Minimized function value / optimized point.
     */

    public static Result minimize_parallel(Fitness fit, double[] lower, double[] upper, double[] guess, int maxEvals,
            double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid, int workers) {
        int dim = fit._dim;
        De opt = new De(lower, upper, guess, popsize, keep, F, CR, Utils.rnd().nextLong(), 0);
        if (workers <= 0 || workers > Threads.numWorkers()) // set default and limit
            workers = Threads.numWorkers();
        Evaluator evaluator = new Evaluator(fit, popsize, workers);
        int evals = 0;
        int stop = 0;    
        for (; evals < maxEvals && stop == 0; evals += workers) {
            double[][] xs = new double[workers][dim];
            int[] pos = new int[workers];
            MutableInt cp = new MutableInt();
            for (int p = 0; p < workers; p++) {
                xs[p] = opt.ask(cp);
                pos[p] = cp.intValue();
            }
            double[] ys = evaluator.eval(xs);
            for (int p = 0; p < workers && stop == 0; p++)
                stop = opt.tell(xs[p], ys[p], pos[p]);
        }
        evaluator.destroy();
        return new Result(fit, fit._evals);
    }

}

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
 *  
 * The wrapped Eigen based implementation of differential evolution uses the DE/best/1 strategy.
    Uses three deviations from the standard DE algorithm:
    a) temporal locality introduced in 
        https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
    b) reinitialization of individuals based on their age.
    c) oscillating CR/F parameters."""
 */

import java.util.Arrays;

import org.apache.commons.lang3.mutable.MutableInt;

import fcmaes.core.Optimizers.Result;

public class De {
	
    private long nativeDe;

    public static Result minimize(Fitness fit, double[] lower, double[] upper, double[] result, int maxEvals,
            double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid, int workers) {
       	if (workers <= 1) {
       		Jni.optimizeDE(fit, lower, upper, result, maxEvals, stopfitness, popsize, keep, F, CR, seed, runid, 1);
    		return new Result(fit, fit._evals);
    	} else {
    		return minimize_parallel(fit, lower, upper, maxEvals, stopfitness, 
    				popsize, keep, F, CR, seed, runid, workers);
    	}
    }

    /**
     * Create a DE object for ask/tell.
     * 
     * @param lower    	lower point limit.
     * @param upper    	upper point limit.
     * @param guess    	Starting point.
     * @param popsize  	Population size used for offspring.
     */

    public De(Fitness fit, double[] lower, double[] upper, int popsize) {
    	int keep = 200; 
    	double F = 0.5;
    	double CR = 0.9;
    	long seed = Utils.rnd().nextLong();
        nativeDe = Jni.initDE(fit, lower, upper, popsize, keep, F, CR, seed, 0);
    }
    
    /**
     * Create a DE object for ask/tell.
     * 
     * @param lower    	lower point limit.
     * @param upper    	upper point limit.
     * @param popsize  	Population size used for offspring.
     * @param keep  	Changes the reinitialization probability of individuals based on their age. 
     * 					Higher value means lower probability of reinitialization.
     * @param F  		The mutation constant. In the literature this is also known as differential weight, 
        		 		being denoted by F. Should be in the range [0, 2].
     * @param CR  		The recombination constant. Should be in the range [0, 1].
     * @param seed  	Random seed.
     * @param runid    	id for debugging/logging.
     */

    public De(Fitness fit, double[] lower, double[] upper, int popsize, double keep, double F, double CR, long seed,
            int runid) {
        nativeDe = Jni.initDE(fit, lower, upper, popsize, keep, F, CR, seed, runid);
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

    public double[] population() {
        return Jni.populationDE(nativeDe);
    }

    /**
     * Minimize evaluating the fitness function in parallel.
     * 
     * @param fit      	fitness function.     
     * @param lower    	lower point limit.
     * @param upper    	upper point limit.
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

	public static Result minimize_parallel(Fitness fit, double[] lower, double[] upper, int maxEvals,
            double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid, int workers) {
		De opt = new De(fit, lower, upper, popsize, keep, F, CR, Utils.rnd().nextLong(), 0);
		int workerLimit = Math.min(popsize, Threads.numWorkers());
		if (workers <= 0 || workers > workerLimit) // set default and limit
			workers = workerLimit;
		Evaluator evaluator = new Evaluator(fit, popsize, workers);
		fit._evals = 0;
		int stop = 0;
		int cp = 0;
		int evals_size = 10*popsize;
		double[][] evals_x = new double[evals_size][];
		int[] evals_p = new int[evals_size];
		// fill eval queue with initial population
		for (int i = 0; i < workers; i++) {
			MutableInt mp = new MutableInt();
			double[] x = opt.ask(mp);
			evaluator.evaluate(x, cp);
			evals_x[cp] = x;
			evals_p[cp] = mp.getValue();
			cp = (cp + 1) % evals_size; 
		}
		while (fit._evals < maxEvals && stop == 0) {
			Evaluator.VecId vid = evaluator.result();
			double y = vid.v[0]; // single objective
			int id = vid.id;
			double[] x = evals_x[id];
			int p = evals_p[id];
			opt.tell(x, y, p);
			MutableInt mp = new MutableInt();
			x = opt.ask(mp);
			evaluator.evaluate(x, cp);
			evals_x[cp] = x;
			evals_p[cp] = mp.getValue();
			cp = (cp + 1) % evals_size; 
		}
		evaluator.join();
		return new Result(fit, fit._evals);
	}

}

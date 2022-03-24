/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

/**
 * Java wrapper for the C++/Eigen based implementation of multi objective
 * Differential Evolution using either DE/rand/1 or DE/best/1 strategy 
 * ('best' refers to the current pareto front').
 *
 * Can switch to NSGA-II like population update via parameter 'nsga_update'.
 * Then it works essentially like NSGA-II but instead of the tournament selection
 * the whole population is sorted and the best individuals survive. To do this
 * efficiently the crowd distance ordering is slightly inaccurate.
 *
 * Supports parallel fitness function evaluation.
 *
 * Enables the comparison of DE and NSGA-II population update mechanism with everything else
 * kept completely identical.
 *
 * Uses the following deviation from the standard DE algorithm:
 * a) oscillating CR/F parameters.
 *
 * You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
 * since the algorithm oscillates between different F and CR settings.
 *
 * For expensive objective functions (e.g. machine learning parameter optimization) use the workers
 * parameter to parallelize objective function evaluation. The workers parameter is limited by the
 * population size.
 */

import java.util.Arrays;

import org.apache.commons.lang3.mutable.MutableInt;

import fcmaes.core.Optimizers.Result;

public class MoDe {
	
    private long nativeMoDe;
    
    /**
     * Determine the pareto front of a multi objective fitness function.
     * 
     * @param fit       multi objective fitness function.
     * @param nobj      number of objectives. 
     * @param ncon      number of constraints.
     * @param lower     lower point limit.
     * @param upper     upper point limit.
     * @param maxEvals  maximal number of fitness function evaluations.
     * @param popsize   Population size used for offspring.
     * @param nsga_update  If true, use NSGA population update, if false, use DE population update.
     * @param pareto_update     DE population update parameter. Only applied if nsga_update = false.
     *                          Only applied if nsga_update = False. Favor better solutions for 
     *                          sample generation. Default 0: use all population members with the same probability.   
     * @param log_period    The log callback is called each log_period iterations.
     * @param workers   Number of parallel threads used to evaluate the fitness function for the population.
     * @return pareto front - flattened argument values for the resulting population.
     */
    public static double[] minimize(FitnessMO fit, int nobj, int ncon, double[] lower, double[] upper, int maxEvals, int popsize,
    		boolean nsga_update, double pareto_update, int log_period, int workers) {
    	int dim = lower.length;    
    	int keep = 200; 
    	double F = 0.5;
    	double CR = 0.9;
    	long seed = Utils.rnd().nextLong();
    	double pro_c = 1.0;
    	double dis_c = 20.0;
    	double pro_m = 1.0; 
    	double dis_m = 20.0;
    	if (workers <= 1)
    		return Jni.optimizeMODE(fit, dim, nobj, ncon, lower, upper, maxEvals, Double.MIN_VALUE, popsize, 
        		keep, F, CR, pro_c, dis_c, pro_m, dis_m, nsga_update, pareto_update, log_period, seed, 0, 0);
    	else
   			return minimize_parallel(fit, nobj, ncon, lower, upper, maxEvals, popsize, 
        		keep, F, CR, pro_c, dis_c, pro_m, dis_m, nsga_update, pareto_update, log_period, workers);
    }
        
    /**
     * @param fit
     * @param nobj
     * @param ncon
     * @param lower
     * @param upper
     * @param maxEvals
     * @param popsize
     * @param keep
     * @param F
     * @param CR
     * @param pro_c
     * @param dis_c
     * @param pro_m
     * @param dis_m
     * @param nsga_update
     * @param pareto_update
     * @param log_period
     * @param workers
     * @return
     */
    public static double[] minimize_parallel(FitnessMO fit, int nobj, int ncon, 
    		double[] lower, double[] upper, int maxEvals, int popsize,
    		double keep, double F, double CR, 
    		double pro_c, double dis_c, double pro_m, double dis_m,
    		boolean nsga_update, double pareto_update, int log_period, int workers) {
        MoDe opt = new MoDe(fit, nobj, ncon, lower, upper, maxEvals, popsize,
        		keep, F, CR, pro_c, dis_c, pro_m, dis_m,
        		nsga_update, pareto_update, Integer.MAX_VALUE, Utils.rnd().nextLong());	
		int workerLimit = Math.min(popsize, Threads.numWorkers());
		if (workers <= 0 || workers > workerLimit) // set default and limit
			workers = workerLimit;
		Evaluator evaluator = new Evaluator(fit, popsize, workers);
		fit._evals = 0;
		int stop = 0;
		double[][] evals_x = new double[popsize][];
		// fill eval queue with initial population
		for (int i = 0; i < workers; i++) {
			MutableInt mp = new MutableInt();
			double[] x = opt.ask(mp);
			evaluator.evaluate(x, mp.getValue());
			evals_x[mp.getValue()] = x;
		}
		while (fit._evals < maxEvals && stop == 0) {
			Evaluator.VecId vid = evaluator.result();
			double[] y = vid.v; // multi objective + constraints
			int p = vid.id;
			double[] x = evals_x[p];
			opt.tell(x, y, p);
			MutableInt mp = new MutableInt();
			x = opt.ask(mp);
			evaluator.evaluate(x, mp.getValue());
			evals_x[mp.getValue()] = x;
		}
		evaluator.join();
		return opt.population();
	}
     
    /**
     * @param fit
     * @param nobj
     * @param ncon
     * @param lower
     * @param upper
     * @param maxEvals
     * @param popsize
     * @param nsga_update
     * @param pareto_update
     * @param log_period
     */
    public MoDe(FitnessMO fit, int nobj, int ncon, double[] lower, double[] upper, int maxEvals, int popsize,
    		boolean nsga_update, double pareto_update, int log_period) {
    	int dim = lower.length;    
    	int keep = 200; 
    	double F = 0.5;
    	double CR = 0.9;
    	long seed = Utils.rnd().nextLong();
    	double pro_c = 1.0;
    	double dis_c = 20.0;
    	double pro_m = 1.0; 
    	double dis_m = 20.0;
    	nativeMoDe = Jni.initMODE(fit, dim, nobj, ncon, lower, upper, maxEvals, Double.MIN_VALUE, popsize, 
        		keep, F, CR, pro_c, dis_c, pro_m, dis_m, nsga_update, pareto_update, log_period, seed, 0);
    }
    
    /**
     * @param nobj
     * @param ncon
     * @param lower
     * @param upper
     * @param maxEvals
     * @param popsize
     * @param keep
     * @param F
     * @param CR
     * @param pro_c
     * @param dis_c
     * @param pro_m
     * @param dis_m
     * @param nsga_update
     * @param pareto_update
     * @param log_period
     * @param seed
     * @param workers
     */
    public MoDe(Fitness fit, int nobj, int ncon, double[] lower, double[] upper, int maxEvals, int popsize, 
    		double keep, double F, double CR, 
    		double pro_c, double dis_c, double pro_m, double dis_m,
    	    boolean nsga_update, double pareto_update, int log_period,
    		long seed) {
    	int dim = lower.length;    	
    	nativeMoDe = Jni.initMODE(fit, dim, nobj, ncon, lower, upper, maxEvals, Double.MIN_VALUE, popsize, 
        		keep, F, CR, pro_c, dis_c, pro_m, dis_m, nsga_update, pareto_update, log_period, seed, 0);
    }
    
    /**
     * Ask for argument vector.
     * 
     * @param pos      	Set to position of argument.     
     * @return  		Argument vector for evaluation.
     */
    public double[] ask(MutableInt pos) {
        double[] asked = Jni.askMODE(nativeMoDe);
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
    public int tell(double[] x, double[] y, int p) {
        return Jni.tellMODE(nativeMoDe, x, y, p);
    }

    /**
     * Destroy corresponding native C++ object 
     * to avoid a memory leak
     */

    public void destroy() {
        Jni.destroyMODE(nativeMoDe);
    }

    public double[] population() {
        return Jni.populationMODE(nativeMoDe);
    }

    protected void finalize() throws Throwable {
        destroy();
    }

}

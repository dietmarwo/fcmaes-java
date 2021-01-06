/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.concurrent.atomic.AtomicInteger;

public class Optimizers {

	public static class Result {
		public int evals;
		public double y;
		public double[] X;

		/**
		 * @param evals     	number of function evaluations.
		 * @param y         	Optimized function value.
		 * @param X     		Optimized point.
		 */
		
		Result(int evals, double y, double[] X) {
			this.evals = evals;
			this.y = y;
			this.X = X;
		}

		/**
		 * @param fit         	Function already optimized.
		 * @param evals     	number of function evaluations.
		 */
		
		Result(Fitness fit, int evals) {
			this.evals = evals;
			this.y = fit._bestY;
			this.X = fit._bestX;
		}
	}

	public static abstract class Optimizer {

		public Optimizer() {
		}

		/**
		 * @param fit         	Function to optimize.
		 * @param lower     	lower point limit.
		 * @param upper     	upper point limit.
		 * @param sigma        	Individual input sigma.
		 * @param guess     	Starting point.
		 * @param maxEvals 		Maximum number of evaluations.
		 * @param stopVal      	Termination criteria for optimization.
		 * @param popsize       Population size used for offspring.
		 * @return Result     	Minimized function value / optimized point.
		 */
		
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {

			throw new RuntimeException("minimize not implemented.");
		}

		/**
		 * Perform a parallel retry. To be used if the objective function is expensive like  
		 * https://ctoc11.skyeststudio.com or https://mintoc.de/index.php/F-8_aircraft
		 * 
		 * @param runs         	number of parallel optimization runs.
		 * @param fit         	Function to optimize.
		 * @param lower     	lower point limit.
		 * @param upper     	upper point limit.
		 * @param sigma        	Individual input sigma.
		 * @param guess     	Starting point.
		 * @param maxEvals 		Maximum number of evaluations.
		 * @param stopVal      	Termination criteria for optimization.
		 * @param popsize       Population size used for offspring.
		 * @return Result     	Minimized function value / optimized point.
		 */
		
		public Result minimizeN(int runs, Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize, double limit) {
			RunOptimizer ropt = new RunOptimizer(runs, this, fit, lower, upper, sigma, guess, maxEvals, stopVal,
					popsize, limit);
			Threads threads = new Threads(ropt);
			threads.start();
			threads.join();
			return new Result(fit, fit._evals);
		}
	}

	/**
	 * Eigen based implementation of active CMA-ES derived from
	 * http:*cma.gforge.inria.fr/cmaes.m which follows
	 * https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review
	 */

	public static class CMA extends Optimizer {

		public CMA() {
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			if (popsize <= 0)
				popsize = 31;
			int evals = Jni.optimizeACMA(fit, lower, upper, sigma, guess, 1000000, maxEvals, stopVal, popsize,
					popsize / 2, 1.0, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}

	}

	/**
	 * Eigen based implementation of differential evolution using on the DE/best/1
	 * strategy. Uses two deviations from the standard DE algorithm: a) temporal
	 * locality introduced in
	 * https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
	 * b) reinitialization of individuals based on their age. requires
	 * https://github.com/imneme/pcg-cpp
     * Doesn't use the sigma (initial stepsize) and guess argument. 
	 */

	public static class DE extends Optimizer {

		public DE() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = dim * 15;
			int evals = Jni.optimizeDE(fit, lower, upper, guess, maxEvals, stopVal, popsize, 200, 0.5, 0.9,
					Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}
	
	/**
     * An experimental new Differential Evolution algorithm derived from DE
     * Doesn't use the sigma (initial stepsize) and guess argument. 
     */
	
	public static class DE2 extends Optimizer {
		
		public DE2() {
			super();
		}
		
		/**
	     * {@inheritDoc}
	     */
		@Override		
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess, 
				  int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0) popsize = dim*15;
//			experimental settings
//			double K1 = Utils.rnd(0.2, 0.7);
//			double K2 = Utils.rnd(0.5, 0.8);
			double K1 = 0;
			double K2 = 0;
			int evals = Jni.optimizeDE2(fit, lower, upper, guess, 
					  maxEvals, stopVal, popsize, 200, K1, K2, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);			
		}		
	}

	/**
	 * Eigen based implementation of differential evolution (GCL-DE) derived from "A
	 * case learning-based differential evolution algorithm for global optimization
	 * of interplanetary trajectory design, Mingcheng Zuo, Guangming Dai, Lei Peng,
	 * Maocai Wang, Zhengquan Liu", https://doi.org/10.1016/j.asoc.2020.106451
     * Doesn't use the sigma (initial stepsize) and guess argument. 
	 */

	public static class GCLDE extends Optimizer {

		public GCLDE() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = (int) (dim * 8.5 + 150);
			int evals = Jni.optimizeGCLDE(fit, lower, upper, guess, maxEvals, stopVal, popsize, 0.7, 0, 0,
					Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
     * An experimental new Differential Evolution algorithm derived from GCLDE
     * Doesn't use the sigma (initial stepsize) and guess argument. 
     */
	
	public static class CLDE extends Optimizer {
		
		public CLDE() {
			super();
		}
		
		/**
	     * {@inheritDoc}
	     */
		@Override		
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess, 
				  int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0) popsize = (int)(dim*8.5+150);
//			experimental settings
//			double K1 = Utils.rnd(0.2, 0.7);
//			double K2 = Utils.rnd(0.5, 0.8);
			double K1 = 0;
			double K2 = 0;
			int evals = Jni.optimizeCLDE(fit, lower, upper, guess, 
					  maxEvals, stopVal, popsize, 0.7, K1, K2, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);			
		}		
	}

	/**
	 * Eigen based implementation of dual annealing derived from
	 * https://github.com/scipy/scipy/blob/master/scipy/optimize/_dual_annealing.py
	 * Implementation only differs regarding boundary handling - this
	 * implementation uses boundary-normalized X values. Local search is fixed to
	 * LBFGS-B, see https://github.com/yixuan/LBFGSpp/tree/master/include.
	 * 
	 * There is an unresolved issue with parallel execution on windows.
	 */
	
	public static class DA extends Optimizer {

		public DA() { // with local optimization
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = dim * 15;
			int evals = Jni.optimizeDA(fit, lower, upper, guess,
					  maxEvals, 1, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}
	
	/**
	 * Like DA, but without local search
	 */
	
	public static class DANL extends Optimizer {

		public DANL() { // no local optimization
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = dim * 15;
			int evals = Jni.optimizeDA(fit, lower, upper, guess,
					maxEvals, 0, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Eigen based implementation of the Harris hawks optimization, see Harris hawks
	 * optimization: Algorithm and applications Ali Asghar Heidari, Seyedali
	 * Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen Future
	 * Generation Computer Systems, DOI:
	 * https://doi.org/10.1016/j.future.2019.02.028
	 * 
	 * derived from
	 * https://github.com/7ossam81/EvoloPy/blob/master/optimizers/HHO.py
	 */
	
	public static class Hawks extends Optimizer {

		public Hawks() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			if (popsize <= 0)
				popsize = 31;
			int evals = Jni.optimizeHawks(fit, lower, upper, guess, maxEvals, stopVal, popsize, Utils.rnd().nextLong(),
					0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Local search around guess derived from DE
	 */
	
	public static class LDE extends Optimizer {

		public LDE() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = dim * 15;
			int evals = Jni.optimizeLDE(fit, lower, upper, guess, sigma, maxEvals, stopVal, popsize, 200, 0.5, 0.9,
					Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}
	
	/**
	 * Local search around guess derived from GCLDE
	 */
	
	public static class LCLDE extends Optimizer {

		public LCLDE() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = (int) (dim * 8.5 + 150);
			int evals = Jni.optimizeLCLDE(fit, lower, upper, guess, sigma, maxEvals, stopVal, popsize, 0.7, 0, 0,
					Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}
	
	/**
	 * Java Mapping of the BITEOPT optimizer, see https://github.com/avaneev/biteopt
	 * Doesn't use the sigma (initial stepsize) argument. 
	 * 	 
	 * derived from
	 * https://github.com/avaneev/biteopt/blob/master/biteopt.h
	 */

	public static class Bite extends Optimizer {

		int M = 1;
		
		public Bite() {
			super();
		}

		public Bite(int M) {
			super();
			this.M = M;
		}

		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int evals = Jni.optimizeBite(fit, lower, upper, guess, maxEvals, stopVal, M, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}
	
	/**
	 * Java Mapping of the CSMA optimizer, see https://github.com/avaneev/biteopt
	 * Similar to CMA-ES, but mainly focuses on sigma adaptation.	
	 * 
	 * Changed to optionally use sigma (initial stepsize), guess, and popsize arguments.
	 *  
	 * derived from
	 * https://github.com/avaneev/biteopt/blob/master/smaesopt.h
	 */
	
	public static class CSMA extends Optimizer {

		public CSMA() {
			super();
		}

		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int evals = Jni.optimizeCsma(fit, lower, upper, sigma, guess, maxEvals, stopVal, popsize, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Sequence of DE and CMA. Evaluations are evenly distributed.
	 */
	
	public static class DECMA extends Optimizer {

		public DECMA() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			if (popsize <= 0)
				popsize = 31;
			int evals = Jni.optimizeDE(fit, lower, upper, guess, maxEvals / 2, stopVal, popsize, 200, 0.5, 0.9,
					Utils.rnd().nextLong(), 0);
			evals += Jni.optimizeACMA(fit, lower, upper, sigma, fit._bestX, 1000000, maxEvals / 2, stopVal, popsize,
					popsize / 2, 1, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Sequence of LDE and CMA. Evaluations are evenly distributed.
	 */
	
	public static class LDECMA extends Optimizer {

		public LDECMA() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			if (popsize <= 0)
				popsize = 31;
			int evals = Jni.optimizeLDE(fit, lower, upper, guess, sigma, maxEvals / 2, stopVal, popsize, 200, 0.5, 0.9,
					Utils.rnd().nextLong(), 0);
			evals += Jni.optimizeACMA(fit, lower, upper, sigma, fit._bestX, 1000000, maxEvals / 2, stopVal, popsize,
					popsize / 2, 1, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Sequence of DE and GCLDE. Evaluations are evenly distributed.
	 */

	public static class DEGCLDE extends Optimizer {

		public DEGCLDE() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = 31;
			int evals = 0;
			if (Utils.rnd().nextBoolean())
				evals += Jni.optimizeDE(fit, lower, upper, guess, maxEvals, stopVal, popsize, 200, 0.5, 0.9,
						Utils.rnd().nextLong(), 0);
			else
				evals += Jni.optimizeGCLDE(fit, lower, upper, guess, maxEvals, stopVal, (int) (dim * 8.5 + 150), 0.7, 0,
						0, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Sequence of DE | GCLDE and CMA. Evaluations are evenly distributed.
	 */

	public static class DEGCLDECMA extends Optimizer {

		public DEGCLDECMA() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = 31;
			int evals = 0;
			if (Utils.rnd().nextBoolean())
				evals += Jni.optimizeDE(fit, lower, upper, guess, maxEvals / 2, stopVal, popsize, 200, 0.5, 0.9,
						Utils.rnd().nextLong(), 0);
			else
				evals += Jni.optimizeGCLDE(fit, lower, upper, guess, maxEvals / 2, stopVal, (int) (dim * 8.5 + 150),
						0.7, 0, 0, Utils.rnd().nextLong(), 0);
			evals += Jni.optimizeACMA(fit, lower, upper, sigma, fit._bestX, 1000000, maxEvals / 2, stopVal, popsize,
					popsize / 2, 1, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	/**
	 * Sequence of LDE | LGCLDE and CMA. Evaluations are evenly distributed.
	 */

	public static class LDELCLDECMA extends Optimizer {

		public LDELCLDECMA() {
			super();
		}

		/**
	     * {@inheritDoc}
	     */
		@Override
		public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
				int maxEvals, double stopVal, int popsize) {
			int dim = guess != null ? guess.length : lower.length;
			if (popsize <= 0)
				popsize = 31;
			int evals = 0;
			if (Utils.rnd().nextBoolean())
				evals += Jni.optimizeLDE(fit, lower, upper, guess, sigma, maxEvals / 2, stopVal, popsize, 200, 0.5,
						0.9, Utils.rnd().nextLong(), 0);
			else
				evals += Jni.optimizeLCLDE(fit, lower, upper, guess, sigma, maxEvals / 2, stopVal,
						(int) (dim * 8.5 + 150), 0.7, 0, 0, Utils.rnd().nextLong(), 0);
			evals += Jni.optimizeACMA(fit, lower, upper, sigma, fit._bestX, 1000000, maxEvals / 2, stopVal, popsize,
					popsize / 2, 1, Utils.rnd().nextLong(), 0);
			return new Result(fit, evals);
		}
	}

	public static class RunOptimizer implements Runnable {
		AtomicInteger count = new AtomicInteger(0);
		int runs;
		Optimizer opt;
		Fitness fit;
		double[] lower;
		double[] upper;
		double[] sigma;
		double[] guess;
		int maxEvals;
		double stopVal;
		int popsize;
		int evals = 0;

		Statistics stat;
		double limit = 0;

		RunOptimizer(int runs, Optimizer opt, Fitness fit, double[] lower, double[] upper, double[] sigma,
				double[] guess, int maxEvals, double stopVal, int popsize, double limit) {

			this.runs = runs;
			this.opt = opt;
			this.fit = fit;
			this.lower = lower;
			this.upper = upper;
			this.sigma = sigma;
			this.guess = guess;
			this.maxEvals = maxEvals;
			this.stopVal = stopVal;
			this.popsize = popsize;
			if (limit != 0) {
				stat = new Statistics();
				this.limit = limit;
			}
		}

		@Override
		public void run() {
			for (;;) {
				int i = count.getAndIncrement();
				if (i >= runs)
					return;
				if (limit != 0) {
					Fitness f = fit.create();
					evals += opt.minimize(f, lower, upper, sigma, guess != null ? guess : Utils.rnd(lower, upper),
							maxEvals, stopVal, popsize).evals;
					if (f._bestY < limit)
						stat.add(f._bestY);
//	                if (i % 100 == 99)
//	                	System.out.println(Utils.r(Utils.measuredMillis()) + " " + i + " " + stat);
					fit.updateBest(f);
				} else {
					evals += opt.minimize(fit, lower, upper, sigma, guess != null ? guess : Utils.rnd(lower, upper),
							maxEvals, stopVal, popsize).evals;
				}
			}
		}
	}
	
}

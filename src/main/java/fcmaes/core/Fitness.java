/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.hipparchus.analysis.UnivariateFunction;
import org.hipparchus.optim.MaxEval;
import org.hipparchus.optim.nonlinear.scalar.GoalType;
import org.hipparchus.optim.univariate.BrentOptimizer;
import org.hipparchus.optim.univariate.MultiStartUnivariateOptimizer;
import org.hipparchus.optim.univariate.SearchInterval;
import org.hipparchus.optim.univariate.UnivariateObjectiveFunction;
import org.hipparchus.optim.univariate.UnivariateOptimizer;
import org.hipparchus.random.RandomGenerator;

import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;

/**
 * Fitness function. Stores the optimization result.
 */

public class Fitness implements Comparable<Fitness>, UnivariateFunction {
				
		/**
		 * Dimension, number of decision variables of the optimization problem.
		 */
		public int _dim;

		/**
		 * Set _stopVal if you want to block function evaluations if _bestY < _stopVal
		 */
		public double _stopVal = Double.NEGATIVE_INFINITY;
		
		/**
		 * Best function value, result of the optimization.
		 */
		public double _bestY = Double.POSITIVE_INFINITY;

		/**
		 * Decision variables corresponding to the best function value, result of the optimization.
		 */
		public double[] _bestX = null;
			
		/**
		 * Number of function evaluations.
		 */
		public int _evals = 0;
		
		/**
		 * Set _parallelEval = true to enable parallel function evaluation for the population.
		 * Set to false for parallel optimization runs.  
		 */
		public boolean _parallelEval = false;
		
		public Fitness(int dim) {
			_dim = dim;
		}

		/**
		 * lower point limit. Needs to be defined for coordinated parallel retry. 
		 */
		public double[] lower() {
			return null;
		}

		/**
		 * upper point limit. Needs to be defined for coordinated parallel retry. 
		 */
		public double[] upper() {
			return null;
		}
		
		/**
		 * Function evaluation. Maps decision variables X to a function value. Overwrite in descendants. 
		 * 
		 * @param X 	Decision variables.
		 * return y 	Function value.
		 */	
		public double eval(double[] X) {
			return Double.NaN;
		}

		/**
		 * Function evaluation wrapper called by the optimization algorithm. 
		 */ 
		public double value(double[] X) {
	        try {
				if (_bestY < _stopVal)
					return _stopVal;
				double y = eval(X);
				_evals++;
				if (y < _bestY) {
					_bestY = y;
					_bestX = X;
				}
				return y;
		    } catch (Exception ex) {
		    	return Double.MAX_VALUE;
		    }
		}

		/**
		 * Function evaluation. Maps a single decision variable to a function value. Overwrite in descendants. 
		 * 
		 * @param x 	Decision variable.
		 * return  		Function value.
		 */	
	    public double value(double x) {
	    	return value(new double[]{x});
	    }
	    
		/**
		 * Clone a fitness function. Used by the parallel coordinated retry. 
		 * Overwrite in descendants. 
		 */ 
	    public Fitness create() {
	    	return null;
	    }

	    public List<Fitness> create(int n) {
	    	List<Fitness> fits = new ArrayList<Fitness>(n);
	    	for (int i = 0; i < n; i++)
	    		fits.add(create());
	    	return fits;
	    }

		/**
		 * Function evaluation. 
		 * Maps decision variables for the whole population to an array of function values. 
		 * Called via JNI from the optimization algorithms. Enables parallel function evaluation. 
		 * 
		 * @param xss 	Decision variables for the whole population.
		 * return 		Function values.
		 */	

		public double[] values(double[] xss) {
			if (_parallelEval)
				return valuesPar(xss);
			else {
		        int popsize = xss.length/_dim;
		        double[] values = new double[popsize];	        
		        for (int i = 0; i < popsize; i++) {
		        	double[] xs = Arrays.copyOfRange(xss, i*_dim, (i+1)*_dim);
		        	values[i] = value(xs);
		        }
		        return values;
			}
	    }
		
		/**
		 * Perform a parallel retry. 
		 * 
		 * @param opt         	Optimizer used.
		 * @param lower     	lower point limit.
		 * @param upper     	upper point limit.
		 * @param guess     	Starting point.
		 * @param sigma        	Individual input sigma.
		 * @param maxEvals 		Maximum number of evaluations.
		 * @param popsize       Population size used for offspring.
		 * @return Result     	Minimized function value / optimized point.
		 */

		public Result minimize(Optimizer opt, double[] lower, double[] upper, 
				double[] guess, double[] sigma, int maxEvals, int popsize) {
			if (guess == null)
				guess = Utils.rnd(lower, upper);
			return opt.minimize(this, lower, upper, sigma, guess, maxEvals, Double.NEGATIVE_INFINITY, popsize);
		}
		
		/**
		 * Perform a single threaded retry. 

		 * @param runs         	number of parallel optimization runs.
		 * @param opt         	Optimizer used.
		 * @param lower     	lower point limit.
		 * @param upper     	upper point limit.
		 * @param guess     	Starting point.
		 * @param sigma        	Individual input sigma.
		 * @param maxEvals 		Maximum number of evaluations.
		 * @param popsize       Population size used for offspring.
		 */

		public void minimizeSer(int runs, Optimizer opt, double[] lower, double[] upper, 
				double[] guess, double[] sigma, int maxEvals, int popsize) {
			for (int i = 0; i < runs; i++) {
				minimize(opt, lower, upper, guess, sigma, maxEvals, popsize);			}
		}

		/**
		 * Perform a parallel retry. 
		 * 
		 * @param runs         	number of parallel optimization runs.
		 * @param opt         	Optimizer used.
		 * @param lower     	lower point limit.
		 * @param upper     	upper point limit.
		 * @param guess     	Starting point.
		 * @param sigma        	Individual input sigma.
		 * @param maxEvals 		Maximum number of evaluations.
		 * @param popsize       Population size used for offspring.
		 * @return Result     	Minimized function value / optimized point.
		 */

		public Result minimizeN(int runs, Optimizer opt, double[] lower, double[] upper, 
				double[] guess, double[] sigma, int maxEvals, int popsize,  double limit) {
			return opt.minimizeN(runs, this, lower, upper, sigma, guess, maxEvals, 
					Double.NEGATIVE_INFINITY, popsize, limit);
		}

		public void minimizeOne(int maxEval) {
			try {
				UnivariateOptimizer opt = new BrentOptimizer(1e-10, 1e-13);
				opt.optimize(new MaxEval(maxEval),
	                    new UnivariateObjectiveFunction(this),
	                    GoalType.MINIMIZE,
	                    new SearchInterval(lower()[0], upper()[0]));
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		}

		public void minimizeOne(int maxEval, int n, RandomGenerator rnd) {
			try {
				UnivariateOptimizer opt = new BrentOptimizer(1e-10, 1e-13);
				opt = new MultiStartUnivariateOptimizer(opt, n, rnd);
				opt.optimize(new MaxEval(maxEval),
	                    new UnivariateObjectiveFunction(this),
	                    GoalType.MINIMIZE,
	                    new SearchInterval(lower()[0], upper()[0]));
			} catch (Exception e) {
			}
		}

		/**
		 * Parallel function evaluation. 
		 * Maps decision variables for the whole population to an array of function values. 
		 * 
		 * @param xss 	Decision variables for the whole population.
		 * return 		Function values.
		 */	
		
		public double[] valuesPar(double[] xss) {
	        int lambda = xss.length/_dim;
	        double[] values = new double[lambda];
	        Values vals = new Values(xss, values, lambda);
	        Threads threads = new Threads(vals);
	        threads.start();
	        threads.join();
	        return values;
	    }
	        
	    private class Values implements Runnable {       
	        AtomicInteger count = new AtomicInteger(0);
	        double[] xss;
	        double[] values;
	        int lambda;
	        
	        Values(double[] xss, double[] values, int lambda) {
	            this.xss = xss;
	            this.values = values;
	            this.lambda = lambda;                      
	        }
	                   
	        @Override
	        public void run() {
	            for (;;) {
	                int i = count.getAndIncrement();
	                if (i >= lambda)
	                    return;
	                double[] xs = Arrays.copyOfRange(xss, i*_dim, (i+1)*_dim);
	                values[i] = value(xs);
	             }
	        }
	    }

		@Override
		public int compareTo(Fitness o) {
			return Double.compare(_bestY, o._bestY);
		}

	}
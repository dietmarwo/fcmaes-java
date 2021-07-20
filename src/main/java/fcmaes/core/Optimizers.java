/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.mutable.MutableInt;

public class Optimizers {

    public static class Result {
        public int evals;
        public double y;
        public double[] X;

        /**
         * @param evals number of function evaluations.
         * @param y     Optimized function value.
         * @param X     Optimized point.
         */

        public Result(int evals, double y, double[] X) {
            this.evals = evals;
            this.y = y;
            this.X = X;
        }

        /**
         * @param fit   Function already optimized.
         * @param evals number of function evaluations.
         */

        public Result(Fitness fit, int evals) {
            this.evals = evals;
            this.y = fit._bestY;
            this.X = fit._bestX;
        }
    }

    public static abstract class Optimizer {

        public Optimizer() {
        }

        /**
         * @param fit      Function to optimize.
         * @param lower    lower point limit.
         * @param upper    upper point limit.
         * @param sigma    Individual input sigma.
         * @param guess    Starting point.
         * @param maxEvals Maximum number of evaluations.
         * @param stopVal  Termination criteria for optimization.
         * @param popsize  Population size used for offspring.
         * @return Result Minimized function value / optimized point.
         */

        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {

            throw new RuntimeException("minimize not implemented.");
        }

        /**
         * Perform a parallel retry. To be used if the objective function is expensive
         * like https://ctoc11.skyeststudio.com or
         * https://mintoc.de/index.php/F-8_aircraft
         * 
         * @param runs     number of parallel optimization runs.
         * @param fit      Function to optimize.
         * @param lower    lower point limit.
         * @param upper    upper point limit.
         * @param sigma    Individual input sigma.
         * @param guess    Starting point.
         * @param maxEvals Maximum number of evaluations.
         * @param stopVal  Termination criteria for optimization.
         * @param popsize  Population size used for offspring.
         * @param limit    only values < limit are used for statistics.
         * @param popsize  Population size used for offspring.
         * @param xs       array of solution vectors for each retry - must be null or of size = runs.
         * @return Result Minimized function value / optimized point.
         */

        public Result minimizeN(int runs, Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, double limit, double[][] xs) {
            RunOptimizer ropt = new RunOptimizer(runs, this, fit, lower, upper, sigma, guess, maxEvals, stopVal,
                    popsize, limit, xs);
            Threads threads = new Threads(ropt);
            threads.start();
            threads.join();
            return new Result(fit, fit._evals);
        }
        
    }

    public static class Bite extends Optimizer {

        int M = 6;
        int stallLimit = 32;

        public Bite() {
            super();
        }

        public Bite(int M) {
            super();
            this.M = M;
        }

        public Bite(int M, int stallLimit) {
            super();
            this.M = M;
            this.stallLimit = stallLimit;
        }

        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            int evals = Jni.optimizeBite(fit, lower, upper, guess, maxEvals, stopVal, M, 
            		stallLimit, Utils.rnd().nextLong(), 0);
            return new Result(fit, evals);
        }
    }

    public static class CSMA extends Optimizer {

        int stallLimit = 32;

        public CSMA() {
            super();
        }
        
        public CSMA(int stallLimit) {
            super();
            this.stallLimit = stallLimit;
        }


        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            int evals = Jni.optimizeCsma(fit, lower, upper, sigma, guess, maxEvals, stopVal, popsize,
            		stallLimit, Utils.rnd().nextLong(), 0);
            return new Result(fit, evals);
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
                int maxEvals, double stopVal, int popsize, int workers) {
            if (popsize <= 0)
                popsize = 31;
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            if (sigma == null)
                sigma = Utils.array(fit._dim, Utils.rnd(0.05, 0.1));
            return Cmaes.minimize(fit, lower, upper, sigma, guess, maxEvals, stopVal, popsize,
                    popsize / 2, 1.0, Utils.rnd().nextLong(), 0, true, -1, workers);
        }

    }
    
    /**
     * Eigen based implementation of active CMA-ES derived from
     * http:*cma.gforge.inria.fr/cmaes.m. Uses the ask / tell interface
     */
   
    public static class CMAAT extends Optimizer {

        public CMAAT() {
            super();
        }

        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {
            if (popsize <= 0)
                popsize = 31;
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            if (sigma == null)
                sigma = Utils.array(fit._dim, Utils.rnd(0.05, 0.1));
            Cmaes cma = new Cmaes(fit, lower, upper, sigma, guess, popsize, popsize / 2, 1.0, Utils.rnd().nextLong(), 0, true,
                    -1);
            int evals = 0;
            int stop = 0;
            int[] p = new int[1];
            for (; evals < maxEvals && stop == 0; evals++) {            	
                double[] x = cma.ask();
                double y = fit.value(x);
                stop = cma.tell(x, y);
            }
            return new Result(fit, evals);
        }
    }
    
    /**
     * Eigen based implementation of differential evolution using on the DE/best/1
     * strategy. Uses two deviations from the standard DE algorithm: a) temporal
     * locality introduced in
     * https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
     * b) reinitialization of individuals based on their age. requires
     * https://github.com/imneme/pcg-cpp Doesn't use the sigma (initial stepsize)
     * and guess argument.
     */

    public static class DE extends Optimizer {

        public DE() {
            super();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] result,
                int maxEvals, double stopVal, int popsize, int workers) {
            if (popsize <= 0)
                popsize = 31;            
            return De.minimize(fit, lower, upper, result, maxEvals, stopVal, popsize, 200, 0.5, 0.9,
                    Utils.rnd().nextLong(), 0, workers);
        }
    }
    
    /**
     * Eigen based implementation of differential evolution using on the DE/best/1
     * strategy. Uses the ask / tell interface
     */
   
    public static class DEAT extends Optimizer {

        public DEAT() {
            super();
        }

        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {
            if (popsize <= 0)
                popsize = 31;
            De de = new De(fit, lower, upper, popsize, 200, 0.5, 0.9, Utils.rnd().nextLong(), 0);
            int evals = 0;
            int stop = 0;
            MutableInt pos = new MutableInt();
            for (; evals < maxEvals && stop == 0; evals++) {
                double[] x = de.ask(pos);
                double y = fit.value(x);
                stop = de.tell(x, y, pos.intValue());
            }
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
                int maxEvals, double stopVal, int popsize, int workers) {
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
                int maxEvals, double stopVal, int popsize, int workers) {
            int dim = guess != null ? guess.length : lower.length;
            if (popsize <= 0)
                popsize = (int) (dim * 8.5 + 150);
//			experimental settings
            double K1 = Utils.rnd(0.2, 0.7);
            double K2 = Utils.rnd(0.5, 0.8);
//			double K1 = 0;
//			double K2 = 0;
            int evals = Jni.optimizeCLDE(fit, lower, upper, guess, maxEvals, stopVal, popsize, 0.7, K1, K2,
                    Utils.rnd().nextLong(), 0);
            return new Result(fit, evals);
        }
    }

    /**
     * Eigen based implementation of dual annealing derived from
     * https://github.com/scipy/scipy/blob/master/scipy/optimize/_dual_annealing.py
     * Implementation only differs regarding boundary handling - this implementation
     * uses boundary-normalized X values. Local search is fixed to LBFGS-B, see
     * https://github.com/yixuan/LBFGSpp/tree/master/include.
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
                int maxEvals, double stopVal, int popsize, int workers) {
            int dim = guess != null ? guess.length : lower.length;
            if (popsize <= 0)
                popsize = dim * 15;
            int evals = Jni.optimizeDA(fit, lower, upper, guess, maxEvals, 1, Utils.rnd().nextLong(), 0);
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
                int maxEvals, double stopVal, int popsize, int workers) {
            int dim = guess != null ? guess.length : lower.length;
            if (popsize <= 0)
                popsize = dim * 15;
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            int evals = Jni.optimizeDA(fit, lower, upper, guess, maxEvals, 0, Utils.rnd().nextLong(), 0);
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
                int maxEvals, double stopVal, int popsize, int workers) {
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
                int maxEvals, double stopVal, int popsize, int workers) {
            int dim = guess != null ? guess.length : lower.length;
            if (popsize <= 0)
                popsize = (int) (dim * 8.5 + 150);
            int evals = Jni.optimizeLCLDE(fit, lower, upper, guess, sigma, maxEvals, stopVal, popsize, 0.7, 0, 0,
                    Utils.rnd().nextLong(), 0);
            return new Result(fit, evals);
        }
    }

    /**
     * Sequence of DE and CMA. Evaluations are randomly distributed.
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
                int maxEvals, double stopVal, int popsize, int workers) {
            if (popsize <= 0)
                popsize = 31;
            if (sigma == null)
                sigma = Utils.array(fit._dim, Utils.rnd(0.05, 0.1));
            double deEvals = Utils.rnd(0.1, 0.5);
            double cmaEvals = 1.0 - deEvals;
            int evals = Jni.optimizeDE(fit, lower, upper, guess, (int) (deEvals * maxEvals), stopVal, popsize, 200, 0.5,
                    0.9, Utils.rnd().nextLong(), 0, 1);
            evals += Jni.optimizeACMA(fit, lower, upper, sigma, fit._bestX, (int) (cmaEvals * maxEvals),
                    stopVal, popsize, popsize / 2, 1, Utils.rnd().nextLong(), 0, true, -1, workers);
            return new Result(fit, evals);
        }
    }

    /**
     * Sequence of Bite and DE. Evaluations are randomly distributed.
     */

    public static class BiteDe extends Optimizer {

        int M = 6;
        int stallLimit = 32;


        public BiteDe() {
            super();
        }

        public BiteDe(int M) {
            super();
            this.M = M;
        }

        public BiteDe(int M, int stallLimit) {
            super();
            this.M = M;
            this.stallLimit = stallLimit;
        }

        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {
            double deEvals = Utils.rnd(0.3, 0.7);
            double biteEvals = 1.0 - deEvals;
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            int evals = Jni.optimizeBite(fit, lower, upper, guess, (int) (biteEvals * maxEvals), stopVal, M, stallLimit,
                    Utils.rnd().nextLong(), 0);
            evals += Jni.optimizeDE(fit, lower, upper, guess, (int) (deEvals * maxEvals), stopVal, popsize, 200, 0.5,
                    0.9, Utils.rnd().nextLong(), 0, 1);
            return new Result(fit, evals);
        }
    }
    
    /**
     * Sequence of DE and Bite. Evaluations are randomly distributed.
     */

    public static class DeBite extends Optimizer {

        int M = 6;
        int stallLimit = 32;

        public DeBite() {
            super();
        }

        public DeBite(int M) {
            super();
            this.M = M;
        }

        public DeBite(int M, int stallLimit) {
            super();
            this.M = M;
            this.stallLimit = stallLimit;
        }

        @Override
        public Result minimize(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
                int maxEvals, double stopVal, int popsize, int workers) {
            double deEvals = Utils.rnd(0.1, 0.5);
            double biteEvals = 1.0 - deEvals;
            if (guess == null)
                guess = Utils.rnd(lower, upper);
            int evals = Jni.optimizeDE(fit, lower, upper, guess, (int) (deEvals * maxEvals), stopVal, popsize, 200, 0.5,
                     0.9, Utils.rnd().nextLong(), 0, 1);
            evals += Jni.optimizeBite(fit, lower, upper, guess, (int) (biteEvals * maxEvals), stopVal, M, 
            		stallLimit, Utils.rnd().nextLong(), 0);
            return new Result(fit, evals);
        }
    }

    /**
     * Sequence of DE | GCLDE and CMA. Evaluations are randomly distributed.
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
                int maxEvals, double stopVal, int popsize, int workers) {
            int dim = guess != null ? guess.length : lower.length;
            if (popsize <= 0)
                popsize = 31;
            int evals = 0;
            double deEvals = Utils.rnd(0.1, 0.5);
            double cmaEvals = 1.0 - deEvals;
            if (Utils.rnd().nextBoolean())
                evals += Jni.optimizeDE(fit, lower, upper, guess, (int) (deEvals * maxEvals), stopVal, 
                        popsize, 200, 0.5, 0.9, Utils.rnd().nextLong(), 0, 1);
            else
                evals += Jni.optimizeGCLDE(fit, lower, upper, guess, (int) (deEvals * maxEvals), stopVal, 
                        (int) (dim * 8.5 + 150), 0.7, 0, 0, Utils.rnd().nextLong(), 0);
            evals += Jni.optimizeACMA(fit, lower, upper, sigma, fit._bestX, (int) (cmaEvals * maxEvals), 
                    stopVal, popsize, popsize / 2, 1, Utils.rnd().nextLong(), 0, true, -1, 1);
            return new Result(fit, evals);
        }
    }
 
    private static class RunOptimizer implements Runnable {
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
        
        double[][] xs;

        RunOptimizer(int runs, Optimizer opt, Fitness fit, double[] lower, double[] upper, double[] sigma,
                double[] guess, int maxEvals, double stopVal, int popsize, double limit, double[][] xs) {

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
            if (xs != null)
                this.xs = xs;
        }

        @Override
        public void run() {
            for (;;) {
                int i = count.getAndIncrement();
                if (i >= runs || (stat != null && stat.getMin() <= stopVal))
                    return;
                if (limit != 0) {
                    Fitness f = fit.create();
                    evals += opt.minimize(f, lower, upper, 
                            sigma != null ? sigma : Utils.array(fit._dim, Utils.rnd(0.05, 0.1)),
                            guess != null ? guess : Utils.rnd(lower, upper),
                            maxEvals, stopVal, popsize, 1).evals;
                    if (f._bestY < limit)
                        stat.add(f._bestY);
                    if (i % 100 == 99 || i == runs-1) {
                        if (f instanceof FitnessMO) {
                        	double[] y = ((FitnessMO)f).moeval(f._bestX);
                        	System.out.println(Utils.r(Utils.measuredMillis()) + " " + (i+1) + 
                        			" " + stat + " " + Arrays.toString(y));
                        } else
                        	System.out.println(Utils.r(Utils.measuredMillis()) + " " + (i+1) + " " + stat);
                    }
                    fit.updateBest(f);
                    if (xs != null)
                        xs[i] = f._bestX;
                } else {
                    evals += opt.minimize(fit, lower, upper, sigma, guess != null ? guess : Utils.rnd(lower, upper),
                            maxEvals, stopVal, popsize, 1).evals;
                }
            }
        }
    }
}

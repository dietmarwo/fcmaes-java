/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.hipparchus.analysis.UnivariateFunction;
import org.hipparchus.optim.MaxEval;
import org.hipparchus.optim.nonlinear.scalar.GoalType;
import org.hipparchus.optim.univariate.BrentOptimizer;
import org.hipparchus.optim.univariate.MultiStartUnivariateOptimizer;
import org.hipparchus.optim.univariate.SearchInterval;
import org.hipparchus.optim.univariate.UnivariateObjectiveFunction;
import org.hipparchus.optim.univariate.UnivariateOptimizer;
import org.hipparchus.random.RandomGenerator;
import org.libj.util.function.TriPredicate;

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
     * Decision variables corresponding to the best function value, result of the
     * optimization.
     */
    public double[] _bestX = null;

    /**
     * Number of function evaluations.
     */
    public int _evals = 0;
   
    /**
     * Creation time.
     */   
    public long _time = System.currentTimeMillis();
    
    /**
     * Callback
     */
    
    public TriPredicate<String,Double,List<Double>> _callBack;
    
    public Fitness(int dim) {
        _dim = dim;
    }

    public boolean callBack(String key, double y, double[] x) {
    	if (_callBack == null)
    		return false;
    	else 
    		return _callBack.test(key, y, Arrays.asList(ArrayUtils.toObject(x)));
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

    public double[] guess() {
        return Utils.rnd(lower(), upper());
    }

    // overwritten by descendants

    /**
     * Function evaluation. Maps decision variables X to a function value. Overwrite
     * in descendants.
     * 
     * @param x Decision variables. return y Function value.
     */
    public double eval(double[] x) {
        return Double.NaN;
    }
    
    // called form the optimization algorithm

    /**
     * Function evaluation wrapper called by the optimization algorithm.
     */
    public double value(double[] x) {
        try {
            if (_bestY < _stopVal)
                return _stopVal;
            double y = eval(x);
            _evals++;
            if (y < _bestY) {
                _bestY = y;
                _bestX = x;
//					System.out.println(_evals + " " + _bestY);
            }
            return y;
        } catch (Exception ex) {
            return Double.MAX_VALUE;
        }
    }
    
    // derived functions

    /**
     * Function evaluation wrapper called by the optimization algorithm.
     */
    public double[] movalue(double[] x) {
        try {
            if (_bestY < _stopVal)
                return new double[] {_stopVal};
            double y = eval(x);
            _evals++;
            if (y < _bestY) {
                _bestY = y;
                _bestX = x;
//					System.out.println(_evals + " " + _bestY);
            }
            return new double[] {y};
        } catch (Exception ex) {
            return null;
        }
    }

    /**
     * Function evaluation. Maps a single decision variable to a function value.
     * Overwrite in descendants.
     * 
     * @param x Decision variable. return Function value.
     */
    public double value(double x) {
        return value(new double[] { x });
    }
 
    /**
     * Print log message from C++
     * 
     * @param s output string.
     */

    public void print(byte[] s) {
        System.out.println(new String(s));
        System.out.flush();
    }

    /**
     * get log values from C++
     */

    public void log(int cols, double[] xdata, double[] ydata) {
        System.out.println(Arrays.toString(ydata));
        System.out.flush();
    }

    /**
     * Clone a fitness function. Used by the parallel coordinated retry. Overwrite
     * in descendants.
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
     * Function evaluation. Maps decision variables for the whole population to an
     * array of function values. Called via JNI from the optimization algorithms.
     * Enables parallel function evaluation.
     * 
     * @param xs Decision variables for the whole population. return Function
     *            values.
     */

    public double[] values(double[] xs) {
        int popsize = xs.length / _dim;
        double[] values = new double[popsize];
        for (int i = 0; i < popsize; i++) {
            double[] x = Arrays.copyOfRange(xs, i * _dim, (i + 1) * _dim);
            values[i] = value(x);
        }
        return values;
    }

    /**
     * Perform a parallel retry.
     * 
     * @param opt      Optimizer used.
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     * @param guess    Starting point.
     * @param sigma    Individual input sigma.
     * @param maxEvals Maximum number of evaluations.
     * @param stopVal  Optimization stops when stopVal is reached.
     * @param popsize  Population size used for offspring.
     * @return Result Minimized function value / optimized point.
     */

    public Result minimize(Optimizer opt, double[] lower, double[] upper, double[] guess, double[] sigma, int maxEvals,
            double stopVal, int popsize) {
        if (guess == null)
            guess = Utils.rnd(lower, upper);
        if (sigma == null)
            sigma = Utils.array(_dim, 0.3);
        return opt.minimize(this, lower, upper, sigma, guess, maxEvals, stopVal, popsize, 1);
    }

    public Result minimize(Optimizer opt, double[] guess, double[] sigma, int maxEvals,
            double stopVal, int popsize) {
        if (guess == null)
            guess = Utils.rnd(lower(), upper());
        if (sigma == null)
            sigma = Utils.array(_dim, 0.3);
        return opt.minimize(this, lower(), upper(), sigma, guess, maxEvals, stopVal, popsize, 1);
    }

    public Result minimizeN(int runs, Optimizer opt, int maxEvals, double stopVal, int popsize, double limit) {
        return opt.minimizeN(runs, this, lower(), upper(), null, null, 
                maxEvals, stopVal, popsize, limit, null);
    }

    public Result minimizeN(int runs, Optimizer opt, double[] guess, int maxEvals, double stopVal, int popsize, double limit) {
        return opt.minimizeN(runs, this, lower(), upper(), null, guess, 
                maxEvals, stopVal, popsize, limit, null);
    }


    /**
     * Perform a single threaded retry.
     * 
     * @param runs     number of parallel optimization runs.
     * @param opt      Optimizer used.
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     * @param guess    Starting point.
     * @param sigma    Individual input sigma.
     * @param maxEvals Maximum number of evaluations.
     * @param stopVal  Optimization stops when stopVal is reached.
     * @param popsize  Population size used for offspring.
     */

    public void minimizeSer(int runs, Optimizer opt, double[] lower, double[] upper, double[] guess, double[] sigma,
            int maxEvals, double stopVal, int popsize) {
        for (int i = 0; i < runs && _bestY > stopVal; i++) {
            minimize(opt, lower, upper, guess, sigma, maxEvals, stopVal, popsize);
        }
    }

    /**
     * Perform a parallel retry.
     * 
     * @param runs     number of parallel optimization runs.
     * @param opt      Optimizer used.
     * @param lower    lower point limit.
     * @param upper    upper point limit.
     * @param guess    Starting point.
     * @param sigma    Individual input sigma.
     * @param maxEvals Maximum number of evaluations.
     * @param stopVal  Optimization stops when stopVal is reached.
     * @param popsize  Population size used for offspring.
     * @return Result Minimized function value / optimized point.
     */

    public Result minimizeN(int runs, Optimizer opt, double[] lower, double[] upper, double[] guess, double[] sigma,
            int maxEvals, double stopVal, int popsize, double limit) {
        if (sigma == null)
            sigma = Utils.array(_dim, 0.3);
        return opt.minimizeN(runs, this, lower, upper, sigma, guess, maxEvals, stopVal, popsize,
                limit, null);
    }

    public void minimizeOne(int maxEval) {
        try {
            UnivariateOptimizer opt = new BrentOptimizer(1e-10, 1e-13);
            opt.optimize(new MaxEval(maxEval), new UnivariateObjectiveFunction(this), GoalType.MINIMIZE,
                    new SearchInterval(lower()[0], upper()[0]));
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public void minimizeOne(int maxEval, int n, RandomGenerator rnd) {
        try {
            UnivariateOptimizer opt = new BrentOptimizer(1e-10, 1e-13);
            opt = new MultiStartUnivariateOptimizer(opt, n, rnd);
            opt.optimize(new MaxEval(maxEval), new UnivariateObjectiveFunction(this), GoalType.MINIMIZE,
                    new SearchInterval(lower()[0], upper()[0]));
        } catch (Exception e) {
        }
    }    
    
    @Override
    public int compareTo(Fitness o) {
        return Double.compare(_bestY, o._bestY);
    }

    void updateBest(Fitness fit) {
        _evals += fit._evals;
        if (fit._bestY < _bestY) {
            _bestY = fit._bestY;
            _bestX = fit._bestX;
        }
    }
}

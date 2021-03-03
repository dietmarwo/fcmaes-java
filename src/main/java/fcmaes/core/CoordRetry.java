/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;

public class CoordRetry {

    /**
     * Perform a coordinated parallel retry. To be used if the objective function is
     * to be evaluated fast but optimization is hard.
     * https://www.esa.int/gsp/ACT/projects/gtop/messenger_full/ can be solved in
     * about one hour on a modern multi core machine.
     * 
     * @param runs       number of parallel optimization runs.
     * @param fit        Function to optimize. Limits - lower() and upper() - have
     *                   to be defined.
     * @param opt        Optimizer used.
     * @param guess      Starting point.
     * @param limitVal   Maximum value for an optimization result to be stored.
     * @param stopVal    Optimization stops when stopVal is reached.
     * @param startEvals Initial maximum number of evaluations. Will increase
     *                   incrementally.
     * @param log        Flag indicating if the results are to be logged.
     */
    public static Result optimize(int runs, Fitness fit, Optimizer opt, double[] guess, double limitVal,
            double stopVal, int startEvals, boolean log) {
        Fitness[] store = new Fitness[500];
        Optimize retry = new Optimize(runs, fit, guess, store, limitVal, stopVal, 
                startEvals, 0, 0, opt, 0, log);
        Threads threads = new Threads(retry);
        threads.start();
        threads.join();
        retry.sort();
        retry.dump(retry._next.get());
        return retry.getResult();
    }

    public static Result optimize(int runs, Fitness fit, Optimizer opt, double[] guess, double limitVal,
            int startEvals, boolean log) {
        return optimize(runs, fit, opt, guess, limitVal, Double.NEGATIVE_INFINITY, startEvals, log);
    }

    public static class Optimize implements Runnable {

        AtomicInteger _next = new AtomicInteger(0);
        double _numRetries = 0;
        double _maxEvals = 0;
        double _maxEvalFac = 0;
        double _evalFacIncr = 0;
        double _evalFac = 1.0;
        int _checkInterval = 0;
        Fitness _fit0;
        double[] _delta;
        double[] _guess;
        Fitness[] _store;
        int _numStored = 0;
        int _numSorted = 0;
        AtomicLong _countAll;
        Optimizer _opt;
        int _popsize;
        double _limitVal;
        double _stopVal;
        double _bestY = Double.POSITIVE_INFINITY;
        double[] _bestX;
        boolean _log;
        Statistics statY = new Statistics();

        /**
         * @param runs          number of parallel optimization runs.
         * @param fit0          Function to optimize. Limits - lower() and upper() -
         *                      have to be defined.
         * @param guess         Starting point.
         * @param store         Store for optimization results used for crossover.
         * @param limitVal      Maximum value for an optimization result to be stored.
         * @param stopVal       Optimization stops when stopVal is reached.
         * @param startEvals    Initial maximum number of evaluations. Will increase
         *                      incrementally.
         * @param maxEvalFac    Maximal evaluation number factor relative to startEval.
         * @param checkInterval Number of optimization runs between sorting the results.
         * @param opt           Optimizer used.
         * @param popsize       Population size used for offspring.
         * @param log           Flag indicating if the results are to be logged.
         */
        public Optimize(int runs, Fitness fit0, double[] guess, Fitness[] store, double limitVal, 
                double stopVal, int startEvals,
                int maxEvalFac, int checkInterval, Optimizer opt, int popsize, boolean log) {
            _limitVal = limitVal;
            _stopVal = stopVal;
            _maxEvals = startEvals > 0 ? startEvals : 1500;
            _maxEvalFac = maxEvalFac > 0 ? maxEvalFac : 50.0;
            _checkInterval = checkInterval > 0 ? checkInterval : 100;
            _numRetries = runs > 0 ? runs : _maxEvalFac * _checkInterval;
            // increment eval_fac so that max_eval_fac is reached at last retry
            _evalFacIncr = _maxEvalFac / (_numRetries / _checkInterval);
            _fit0 = fit0;
            _store = store;
            _opt = opt;
            _popsize = popsize;
            _countAll = new AtomicLong(0);
            _guess = guess;
            _delta = Utils.minus(_fit0.upper(), _fit0.lower());
            _log = log;
        }

        public Result getResult() {
            return new Result(_countAll.intValue(), _bestY, _bestX);
        }

        public void dump(int iter) {
            if (_log)
                synchronized (_store) {
                    if (_numStored > 0) {
                        double best = _bestY;
                        double worst = _store[_numStored - 1]._bestY;
                        double time = Utils.measuredSeconds();
                        System.out.println(iter + " " + _numStored + " " + Utils.r(_evalFac, 1) + " " + Utils.r(time)
                                + " " + _countAll.get() + " " + Utils.r((_countAll.get() + 0.0) / time) + " "
                                + Utils.r(best, 8) + " " + Utils.r(worst) + " " + statY.toString() + " " + storeVals()
                                + " " + Arrays.toString(_bestX));
                    }
                }
        }

        @Override
        public void run() {
            int i;
            while ((i = _next.getAndIncrement()) < _numRetries && statY.getMin() >= _stopVal) {
                if (crossover(i))
                    continue;
                Fitness fit = _fit0.create();
                double[] sdev = Utils.array(_fit0._dim, Utils.rnd(0.05, 0.1));
                fit.minimize(_opt, _fit0.lower(), _fit0.upper(), _guess, sdev, evalNum(), _stopVal, _popsize);
                addResult(i, fit, _limitVal);
            }
        }

        private void addResult(int countRuns, Fitness fit, double limit) {
            synchronized (_store) {
                incrCountEvals(countRuns, fit._evals);
                if (fit._bestY < limit) {
                    statY.add(fit._bestY);
                    if (fit._bestY < _bestY) {
                        _bestY = fit._bestY;
                        _bestX = fit._bestX;
                        dump(countRuns);
                    }
                    if (_numStored >= _store.length - 1)
                        sort();
                    _store[_numStored++] = fit;
                }
            }
        }

        /**
         * Crossover of optimization results in the store. Better results have a higher
         * chance to be chosen.
         */
        private boolean crossover(int countRuns) {
            if (Utils.rnd().nextBoolean() || _numSorted < 2)
                return false;
            Fitness fit1 = null;
            Fitness fit2 = null;
            synchronized (_store) {
                int i1 = -1;
                int i2 = -1;
                int n = _numSorted;
                double lim = Utils.rnd(Math.min(0.1 * n, 1.0), 0.2 * n) / n;
                outer: for (int i = 0; i < 100; i++) {
                    i1 = -1;
                    i2 = -1;
                    for (int j = 0; j < n; j++) {
                        if (Utils.rnd(lim)) {
                            if (i1 < 0)
                                i1 = j;
                            else {
                                i2 = j;
                                break outer;
                            }
                        }
                    }
                }
                if (i2 < 0)
                    return false;
                fit1 = _store[i1];
                fit2 = _store[i2];
            }
            double[] x0 = fit1._bestX;
            double[] x1 = fit2._bestX;

            double diffFac = Utils.rnd(0.5, 1.0);
            double limFac = Utils.rnd(2.0, 4.0) * diffFac;

            double[] deltax = Utils.minusAbs(x1, x0);
            double[] delta_bound = Utils.maximum(Utils.sprod(deltax, limFac), 0.0001);
            Fitness fit = _fit0.create();
            double[] lower = Utils.maximum(_fit0.lower(), Utils.minus(x0, delta_bound));
            double[] upper = Utils.minimum(_fit0.upper(), Utils.plus(x0, delta_bound));
            double[] guess = Utils.fitting(x1, lower, upper);
            double[] sdev = Utils.fitting(Utils.sprod(Utils.quot(deltax, _delta), diffFac), 0.001, 0.5);
            fit.minimize(_opt, lower, upper, guess, sdev, evalNum(), _stopVal, _popsize);
            addResult(countRuns, fit, fit1._bestY);
            return true;
        }

        private int evalNum() {
            return (int) (_evalFac * _maxEvals);
        }

        /**
         * Registers the number of evaluations of an optimization run. Triggers sorting
         * after check_interval calls.
         */
        private void incrCountEvals(int countRuns, int evals) {
            if (countRuns % _checkInterval == _checkInterval - 1) {
                if (_evalFac < _maxEvalFac)
                    _evalFac += _evalFacIncr;
                sort();
            }
            _countAll.addAndGet(evals);
        }

        private void sort() {
            Fitness[] sorted = _store.clone();
            Arrays.sort(sorted, 0, _numStored);
            Fitness prev = null;
            Fitness prev2 = null;
            int j = 0;
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = sorted[i];

                if ((prev == null || distance(prev._bestX, fit._bestX) > 0.15)
                        && (prev2 == null || distance(prev2._bestX, fit._bestX) > 0.15)) {
                    _store[j++] = fit;
                    prev2 = prev;
                    prev = fit;
                }
            }
            _numSorted = _numStored = Math.min(j, (int) (0.9 * _store.length));
        }

        /**
         * normalized distance between entries in the store.
         */
        private double distance(double[] xp, double[] x) {
            //
            return Utils.norm(Utils.quot(Utils.minus(x, xp), _delta)) / Math.sqrt(_fit0._dim);
        }

        private String storeVals() {
            StringBuffer buf = new StringBuffer();
            buf.append("[");
            for (int i = 0; i < 20; i++) {
                Fitness fit = _store[i];
                if (fit == null)
                    break;
                if (i > 0)
                    buf.append(", ");
                buf.append(Utils.r(fit._bestY, 1));
            }
            buf.append("]");
            return buf.toString();
        }

    }

}

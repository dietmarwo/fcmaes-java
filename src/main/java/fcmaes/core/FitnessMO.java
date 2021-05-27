/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.io.FileNotFoundException;
import java.util.Arrays;

import fcmaes.core.Optimizers.Optimizer;

/**
 * Fitness function. Stores the optimization result.
 */

public class FitnessMO extends Fitness {
    
    public FitnessMO(int dim, double[] weights, double exp) {
        super(dim);
        _nobj = weights.length;
        _weights = weights;
        _exp = exp;
    }

    public FitnessMO(int dim, double[] lower_weights, double[] upper_weights, double exp) {
        super(dim);
        _exp = exp;
        _nobj = lower_weights.length;
        _lower_weights = lower_weights;
        _upper_weights = upper_weights;
        _weights = Utils.rnd(0, 1, _upper_weights.length);
        _weights = Utils.sprod(_weights, 1.0 / avg_exp(_weights));// correct scaling    
        _weights = Utils.plus(lower_weights, Utils.prod(_weights, 
        		Utils.minus(upper_weights, lower_weights)));
    }

    /**
     * number of objectives.
     */
    
    public int _nobj;

    public double _exp;

    public double[] _weights;

    public double[] _lower_weights;

    public double[] _upper_weights;

    /**
     * Function evaluation. Maps decision variables X to a value vector. Overwrite
     * in descendants.
     * @param X Decision variables. 
     * @return Y value vector.
     */
    public double[] moeval(double[] x) {
        return null;
    }

    /**
     * Single objective mapping using weighted sum. 
     * 
     * @param x Decision variables. 
     * @return y Function value.
     */
    public double eval(double[] x) {        
        double[] y = Utils.prod(_weights, moeval(x));
        return avg_exp(y);
    }

    public double[][] moevals(double[][] xs) {
        double[][] ys = new double[xs.length][];
        for (int i = 0; i < xs.length; i++)
            ys[i] = moeval(xs[i]);
        return ys;
    }
    
    public double[][] minimizeMO(int runs, Optimizer opt, int maxEvals, double stopVal, int popsize, double limit) {
        double[][] xs = new double[runs][];
        opt.minimizeN(runs, this, lower(), upper(), null, null, 
                maxEvals, stopVal, popsize, limit == 0 ? Double.MAX_VALUE : limit, xs);
        return xs;
    }
 
    public double[][] pareto_front(double ys[][]) {
        boolean[] filter = pareto_filter(ys);
        return filter(ys, filter);
    }

    public boolean[] pareto_filter(double ys[][]) {
        boolean[] pareto = new boolean[ys.length];
        for (int index = 0; index < ys.length; index++) {
            pareto[index] = true;
            for (int i = 0; i < ys.length; i++)
                if (i != index && allLess(ys[i], ys[index])) {
                    pareto[index] = false;
                    break;
                }
        }
        return pareto;
    }
           
    public double[][] filter(double ys[][], boolean[] filter) {
        double[][] filtered = new double[ys.length][];
        int j = 0;
        for (int i = 0; i < ys.length; i++)
            if (filter[i])
                filtered[j++] = ys[i];
        return Arrays.copyOfRange(filtered, 0, j);
    }
       
    private double avg_exp(double[] y) {
        double sum = 0;
        for (int i = 0; i < y.length; i++)
            sum += pow(y[i], _exp);
        return pow(sum, 1.0/_exp);
    }
    
    private static double pow(double x, double y) {
        return x > 0 ? Math.pow(x, y) : -Math.pow(-x, y);
    }
    
    private static boolean allLess(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++)
            if (a[i] >= b[i])
                return false;
        return true;
    }
        
    public static void main(String[] args) throws FileNotFoundException {
        System.out.println(pow(-2, 0.5));
        System.out.println(pow(-2, 2));   
        System.out.println(pow(-2, 3));   
    }

}

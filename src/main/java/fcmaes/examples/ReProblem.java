/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.examples;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import com.nativeutils.NativeUtils;

import fcmaes.core.FitnessMO;
import fcmaes.core.MoDe;
import fcmaes.core.JFPlot;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.CMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Utils;

/**
 * Java wrapper for https://github.com/ryojitanabe/reproblems/blob/master/reproblem_c_ver/reproblem.c
 * A real-world multi-objective problem suite (the RE benchmark set) 
 * Reference: 
 * Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite" (submitted) 
 *  Copyright (c) 2018 Ryoji Tanabe 
 */


public class ReProblem extends FitnessMO {

   	// force loading of shared libary
	static boolean loaded = fcmaes.core.Jni.libraryLoaded;

    String _problem;    	
	int _nconstr;       
	double[] _lower;
	double[] _upper;
    
    public ReProblem(String problem, double[] lower_weights, double[] upper_weights, double exp) {
        super(0, lower_weights, upper_weights, exp);
        init(problem);
    }

    void init(String problem) {
     	_problem = problem;
    	double[] problemData = Jni.bounds_re_C(_problem);
    	if (problemData.length == 0)
    		throw new RuntimeException("Unknown REProblem " + problem);
    	int j = 0;
    	_dim = (int) problemData[j++];
    	_nobj = (int) problemData[j++];    	
    	_nconstr = (int) problemData[j++];       
    	_lower = new double[_dim];
    	_upper = new double[_dim];
    	for (int i = 0; i < _dim; i++)
    		_lower[i] = problemData[j++]; 
    	for (int i = 0; i < _dim; i++)
    		_upper[i] = problemData[j++]; 
    }
    
    @Override
    public double[] lower() {
        return _lower;
    }

    @Override
    public double[] upper() {
        return _upper;
    }
   
    public double[] moeval(double[] x) {
        double[] y = Jni.objectives_re_C(_problem, x);
        return y;
    }

    public ReProblem create() {
        return new ReProblem(_problem, _lower_weights, _upper_weights, _exp);
    }
    
    double[][] optimize(int num, Optimizer opt, int maxEvals) {
        return minimizeMO(num, opt, maxEvals, -1E99, 31, 1E99); 
    }
    
    /**
     * Solve the multi objective problem RE21 by parallel single objective 
     * retry applying random weights
     */
    static void test1() {
	    Utils.startTiming();
	    Optimizer opt = new CMA();
	//    Optimizer opt = new DECMA();
	    //Optimizer opt = new Bite(16);
	    ReProblem reProb = new ReProblem("RE21", 
	           new double[] {0, 10}, new double[] {0.001, 100}, 1.0);
	    double[][] xs = reProb.optimize(3200, opt, 1000);
	    double[][] ys = reProb.moevals(xs);
	    boolean[] filter = new boolean[ys.length];
	    for (int i = 0; i < ys.length; i++)
	    	filter[i] = !Utils.isNaN(ys[i]);
	    ys = reProb.filter(ys, filter);
	    double[][] yp = reProb.pareto_front(ys);
	    JFPlot jf1 = new JFPlot(ys, 1000, 1000);
	    jf1.writeAsImage("yall1");
	    JFPlot jf2 = new JFPlot(yp, 1000, 1000);
	    jf2.writeAsImage("yfront1");
    }

    /**
     * Solve the multi objective problem RE21 by the new MoDe algorithm applying NSGA
     * population update
     */
    static void test2() {
	    Utils.startTiming();
	    ReProblem reProb = new ReProblem("RE21", 
	           new double[] {0, 10}, new double[] {0.001, 100}, 1.0);	    
	    double[] xss = MoDe.minimize(reProb, reProb._nobj, reProb._nconstr, 
	    		reProb.lower(), reProb.upper(), 100000, 128, 
	    		false, 0, Integer.MAX_VALUE, 32);
	    int dim = reProb._dim;
	    int n = xss.length/dim;
	    double[][] xs = new double[n][];
	    for (int i = 0; i < n; i++)
	    	xs[i] = Arrays.copyOfRange(xss, i*dim, (i+1)*dim);
	    double[][] ys = reProb.moevals(xs);
	    boolean[] filter = new boolean[ys.length];
	    for (int i = 0; i < ys.length; i++)
	    	filter[i] = !Utils.isNaN(ys[i]);
	    ys = reProb.filter(ys, filter);
	    double[][] yp = reProb.pareto_front(ys);
	    JFPlot jf1 = new JFPlot(ys, 1000, 1000);
	    jf1.writeAsImage("yall2");
	    JFPlot jf2 = new JFPlot(yp, 1000, 1000);
	    jf2.writeAsImage("yfront2");
    }

    public static void main(String[] args) throws FileNotFoundException {
	    Log.setLog();
	    test1();
	    test2();
    }
}

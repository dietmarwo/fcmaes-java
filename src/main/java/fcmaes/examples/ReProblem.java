/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.examples;

import java.io.FileNotFoundException;

import fcmaes.core.FitnessMO;
import fcmaes.core.JFPlot;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.CMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Utils;

public class ReProblem extends FitnessMO {

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
    
    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new CMA();
        //Optimizer opt = new DECMA();
        //Optimizer opt = new Bite(16);
        ReProblem reProb = new ReProblem("RE21", 
               new double[] {0, 10}, new double[] {0.001, 100}, 1.0);
        double[][] xs = reProb.optimize(3200, opt, 1000);
        double[][] ys = reProb.moevals(xs);
        double[][] yp = reProb.pareto_front(ys);
        JFPlot jf1 = new JFPlot(ys, 1000, 1000);
        jf1.writeAsImage("yall2");
        JFPlot jf2 = new JFPlot(yp, 1000, 1000);
        jf2.writeAsImage("yfront2");
        
    }

}

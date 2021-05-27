/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.examples;

import java.io.FileNotFoundException;

import fcmaes.core.Fitness;
import fcmaes.core.FitnessMO;
import fcmaes.core.JFPlot;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.Bite;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Utils;

public class CassiniMO extends FitnessMO {

    Fitness _base;
    
    public CassiniMO(Fitness base, double[] lower_weights, double[] upper_weights, double exp) {
        super(base._dim, lower_weights, upper_weights, exp);
        _base = base;
    }

    public double[] moeval(double[] x) {
        double y = _base.eval(x);
        double tf = 0;
        for (int i = 1; i < x.length; i++)
            tf += x[i];
        return new double[] {y, tf};
    }
    
    @Override
    public double[] lower() {
        return _base.lower();
    }

    @Override
    public double[] upper() {
        return _base.upper();
    }


    public CassiniMO create() {
        return new CassiniMO(_base, _lower_weights, _upper_weights, _exp);
    }
    
    double[][] optimize(int num, Optimizer opt, int maxEvals) {
        return minimizeMO(num, opt, maxEvals, -1E99, 31, 1E99); 
    }


    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        //Optimizer opt = new Bite(16);
        CassiniMO cassMO = new CassiniMO(new Cassini1(), 
               new double[] {1, 0.01}, new double[] {100, 1}, 2.0);
        double[][] xs = cassMO.optimize(5000, opt, 50000);
        double[][] ys = cassMO.moevals(xs);
        boolean[] filter = new boolean[ys.length];
        for (int i = 0; i < ys.length; i++)
            filter[i] = ys[i][0] < 40 ;//&& ys[i][1] < 2500;
        ys = cassMO.filter(ys, filter);
        double[][] yp = cassMO.pareto_front(ys);
        JFPlot jf1 = new JFPlot(ys, 1000, 1000);
        jf1.writeAsImage("yall2");
        JFPlot jf2 = new JFPlot(yp, 1000, 1000);
        jf2.writeAsImage("yfront2");
        
    }

}

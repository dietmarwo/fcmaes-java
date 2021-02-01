package fcmaes.examples;

import java.io.FileNotFoundException;
import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Gtoc1 extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 15 seconds 
     * on a modern 16-core CPU (AMD 5950x).
     * Can you provide a faster parallel algorithm in any language?
     */

    @Override
    public double[] lower() {
        return new double[] { 3000., 14., 14., 14., 14., 100., 366., 300. };
    }

    @Override
    public double[] upper() {
        return new double[] { 10000., 2000., 2000., 2000., 2000., 9000., 9000., 9000. };
    }

    public Gtoc1() {
        super(8);
    }

    public Gtoc1 create() {
        return new Gtoc1();
    }

    @Override
    public double eval(double[] point) {
        try {
            int[] seq = new int[_dim];
            int[] rev = new int[_dim];
            double dvLaunch = 0;
            double[] rp = new double[_dim];
            double[] dv = new double[_dim];
            return Jni.gtoc1_C(point, seq, rev, dvLaunch, rp, dv) - 2000000;
        } catch (Exception ex) {
            return 1E10;
        }
    }
    
    double limitVal() {
        return -300000;
    }

    double stopVal() {
        return -1581950.0 / stopValFac();
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new Gtoc1().test(100, opt, 10000);
    }

}

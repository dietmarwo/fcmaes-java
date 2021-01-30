package fcmaes.examples;

import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Gtoc1 extends Fitness {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord()" solves the problem in less than 60 seconds
     * on a modern 16-core CPU. 
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

    static Result optimize() {
        Optimizer opt = new DECMA();
        Gtoc1 fit = new Gtoc1();
        double[] sdev = Utils.array(fit._dim, 0.07);
        return fit.minimizeN(10000, opt, fit.lower(), fit.upper(), null, sdev, 10000, 31, 10000000.0);
    }
    
    static Result coord() {
        return CoordRetry.optimize(4000, new Gtoc1(), new DECMA(), null, 0, 1500, true);
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        Utils.startTiming();
//        Result res = optimize();
        Result res = coord();
        System.out.println("best = " + res.y 
                + ", time = " + 0.001*Utils.measuredMillis() + " sec, evals = " + res.evals);

    }

}

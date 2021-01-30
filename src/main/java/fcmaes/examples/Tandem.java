package fcmaes.examples;

import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.DEGCLDECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Tandem extends Fitness {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord()" solves the problem often in less than 400 seconds
     * on a modern 16-core CPU. 
     * Can you provide a faster parallel algorithm in any language?
     * 
     * Note that Tandem is a problem were DEGCLDECMA() could be 
     * stronger than DECMA() (but slightly slower) for coordinated retry. 
     */

    @Override
    public double[] lower() {
        return new double[] { 5475, 2.5, 0, 0, 20, 20, 20, 20, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, -Math.PI,
                -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 9132, 4.9, 1, 1, 2500, 2500, 2500, 2500, 0.99, 0.99, 0.99, 0.99, 10, 10, 10, Math.PI,
                Math.PI, Math.PI };
    }

    public Tandem() {
        super(18);
    }

    public Tandem create() {
        return new Tandem();
    }

    @Override
    public double eval(double[] point) {
        try {
            int[] seq = new int[] { 3, 2, 3, 3, 6 };
            return Jni.tandem_C(point, seq);
        } catch (Exception ex) {
            return 1E10;
        }
    }

    static Result optimize() {
        Optimizer opt = new DECMA();
        Tandem fit = new Tandem();
        double[] sdev = Utils.array(fit._dim, 0.07);
        return fit.minimizeN(10000, opt, fit.lower(), fit.upper(), null, sdev, 10000, 31, 100.0);
    }

    static Result coord() {
//        return CoordRetry.optimize(20000, new Tandem(), new DEGCLDECMA(), null, -300.0, 1500, true);
        return CoordRetry.optimize(20000, new Tandem(), new DECMA(), null, -300.0, 1500, true);
    }
 
    public static void main(String[] args) throws NumberFormatException, IOException {
        Utils.startTiming();
//      Result res = optimize();
        Result res = coord();
        System.out.println(
                "best = " + res.y + ", time = " + 0.001 * Utils.measuredMillis() + " sec, evals = " + res.evals);
    }

}

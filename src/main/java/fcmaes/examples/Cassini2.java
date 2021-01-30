package fcmaes.examples;

import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Cassini2 extends Fitness {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord()" solves the problem in less than 70 seconds
     * on a modern 16-core CPU. 
     * Can you provide a faster parallel algorithm in any language?
     */

    @Override
    public double[] lower() {
        return new double[] { -1000, 3, 0, 0, 100, 100, 30, 400, 800, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.15,
                1.7, -Math.PI, -Math.PI, -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 0, 5, 1, 1, 400, 500, 300, 1600, 2200, 0.9, 0.9, 0.9, 0.9, 0.9, 6, 6, 6.5, 291, Math.PI,
                Math.PI, Math.PI, Math.PI };
    }

    public Cassini2() {
        super(22);
    }

    public Cassini2 create() {
        return new Cassini2();
    }

    @Override
    public double eval(double[] point) {
        try {
            return Jni.cassini2_C(point);
        } catch (Exception ex) {
            return 1E10;
        }
    }

    static Result optimize() {
        Optimizer opt = new DECMA();
        Cassini2 fit = new Cassini2();
        double[] sdev = Utils.array(fit._dim, 0.07);
        return fit.minimizeN(10000, opt, fit.lower(), fit.upper(), null, sdev, 50000, 31, 100.0);
    }

    static Result coord() {
        return CoordRetry.optimize(3000, new Cassini2(), new DECMA(), null, 20.0, 1500, true);
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        Utils.startTiming();
//      Result res = optimize();
        Result res = coord();
        System.out.println(
                "best = " + res.y + ", time = " + 0.001 * Utils.measuredMillis() + " sec, evals = " + res.evals);
    }
}

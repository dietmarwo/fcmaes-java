package fcmaes.examples;

import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Cassini1 extends Fitness {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord()" solves the problem in less than 20 seconds
     * on a modern 16-core CPU. 
     * Can you provide a faster parallel algorithm in any language?
     */

    double[] _rpBest;

    @Override
    public double[] lower() {
        return new double[] { -1000, 30, 100, 30, 400, 1000 };
    }

    @Override
    public double[] upper() {
        return new double[] { 0, 400, 470, 400, 2000, 6000 };
    }

    public Cassini1() {
        super(6);
    }

    public Cassini1 create() {
        return new Cassini1();
    }

    @Override
    public double eval(double[] point) {
        try {
            double[] rp = new double[6];
            double value = Jni.cassini1_C(point, rp);
            if (value < _bestY)
                _rpBest = rp;
            return value;
        } catch (Exception ex) {
            return 1E10;
        }
    }

    static Result optimize() {
        Optimizer opt = new DECMA();
        Cassini1 fit = new Cassini1();
        double[] sdev = Utils.array(fit._dim, 0.07);
        return fit.minimizeN(10000, opt, fit.lower(), fit.upper(), null, sdev, 10000, 31, 12.0);
    }

    static Result coord() {
        return CoordRetry.optimize(2000, new Cassini1(), new DECMA(), null, 20.0, 1500, true);
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        Utils.startTiming();
//      Result res = optimize();
        Result res = coord();
        System.out.println(
                "best = " + res.y + ", time = " + 0.001 * Utils.measuredMillis() + " sec, evals = " + res.evals);
    }

}

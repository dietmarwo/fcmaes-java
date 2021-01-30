package fcmaes.examples;

import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class Messenger extends Fitness {

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
        return new double[] { 1000., 1., 0., 0., 200., 30., 30., 30., 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.1, -Math.PI,
                -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 4000., 5., 1., 1., 400., 400., 400., 400., 0.99, 0.99, 0.99, 0.99, 6, 6, 6, Math.PI,
                Math.PI, Math.PI };
    }

    public Messenger() {
        super(18);
    }

    public Messenger create() {
        return new Messenger();
    }

    @Override
    public double eval(double[] point) {
        try {
            return Jni.messenger_C(point);
        } catch (Exception ex) {
            return 1E10;
        }
    }

    static Result optimize() {
        Optimizer opt = new DECMA();
        Messenger fit = new Messenger();
        double[] sdev = Utils.array(fit._dim, 0.07);
        return fit.minimizeN(10000, opt, fit.lower(), fit.upper(), null, sdev, 10000, 31, 100.0);
    }

    static Result coord() {
        return CoordRetry.optimize(4000, new Messenger(), new DECMA(), null, 20.0, 1500, true);
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        Utils.startTiming();
//      Result res = optimize();
        Result res = coord();
        System.out.println(
                "best = " + res.y + ", time = " + 0.001 * Utils.measuredMillis() + " sec, evals = " + res.evals);
    }

}

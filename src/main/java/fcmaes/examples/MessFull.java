package fcmaes.examples;

import java.io.IOException;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;

public class MessFull extends Fitness {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord()" solves the problem often in less than 1300 seconds
     * on a modern 16-core CPU (< 1000 sec using an AMD 5950x).
     * Can you provide a faster parallel algorithm in any language?
     * 
     * The Messenger Full problem ist the hardest of the GTOP problems, 
     * the only one beside Tandem which isn't solved in every coord() run. Sometimes
     * the algorithm is stuck at a local minimum around 2.4. 
     * The algorithm described in http://www.midaco-solver.com/data/pub/PDPTA20_Messenger.pdf
     * hasn't this problem, but requires a 1000 CPU node cluster. 
     */

    @Override
    public double[] lower() {
        return new double[] { 1900.0, 3.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -Math.PI, -Math.PI, -Math.PI, -Math.PI, -Math.PI };
    }

    @Override
    public double[] upper() {
        return new double[] { 2200.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 550.0, 0.99, 0.99, 0.99, 0.99,
                0.99, 0.99, 6.0, 6.0, 6.0, 6.0, 6.0, Math.PI, Math.PI, Math.PI, Math.PI, Math.PI };
    }

    public MessFull() {
        super(26);
    }

    public MessFull create() {
        return new MessFull();
    }

    @Override
    public double eval(double[] point) {
        try {
            return Jni.messengerfull_C(point);
        } catch (Exception ex) {
            return 1E10;
        }
    }

    static Result optimize() {
        Optimizer opt = new DECMA();
        MessFull fit = new MessFull();
        double[] sdev = Utils.array(fit._dim, 0.07);
        return fit.minimizeN(10000, opt, fit.lower(), fit.upper(), null, sdev, 10000, 31, 100.0);
    }
 
    static Result coord() {
        for (int i = 0; i < 10000000; i++) {
            Utils.startTiming();
            CoordRetry.optimize(50000, new MessFull(), new DECMA(), null, 12.0, 1500, true);
        }
        return CoordRetry.optimize(50000, new MessFull(), new DECMA(), null, 12.0, 1500, true);
    }

    public static void main(String[] args) throws NumberFormatException, IOException {
        Utils.startTiming();
//      Result res = optimize();
        Result res = coord();
        System.out.println(
                "best = " + res.y + ", time = " + 0.001 * Utils.measuredMillis() + " sec, evals = " + res.evals);
    }

}

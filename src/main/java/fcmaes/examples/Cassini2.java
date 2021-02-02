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

public class Cassini2 extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 20 seconds 
     * on a modern 16-core CPU (AMD 5950x).
     * Can you provide a faster parallel algorithm?
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

    double limitVal() {
        return 20.0;
    }
 
    double stopVal() {
        return stopValFac()*8.3830;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new Cassini2().test(100, opt, 6000);
    }
}

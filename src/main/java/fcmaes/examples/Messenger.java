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

public class Messenger extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 11 seconds 
     * on a modern 16-core CPU (AMD 5950x).
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

    double limitVal() {
        return 20.0;
    }

    double stopVal() {
        return stopValFac()*8.6299;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new Messenger().test(100, opt, 8000);
    }

}

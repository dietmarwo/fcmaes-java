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

public class Cassini1 extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 1.6 seconds 
     * on a modern 16-core CPU (AMD 5950x).
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

    double limitVal() {
        return 20.0;
    }
 
    double stopVal() {
        return stopValFac()*4.9307;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new Cassini1().test(100, opt, 4000);
    }

}

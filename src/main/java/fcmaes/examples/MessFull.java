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

public class MessFull extends GtopProblem {

    /*
     * This example is taken from https://www.esa.int/gsp/ACT/projects/gtop/
     * See also http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
     * where you find implementations in different programming languages. 
     * 
     * "coord(new DECMA(), runs))" solves the problem on average in about 2250 seconds 
     * on a modern 16-core CPU (AMD 5950x). A value of 2.0 is reached in about 1250 seconds.
     * Can you provide a faster parallel algorithm?
     * 
     * The Messenger Full problem is the hardest of the GTOP problems, 
     * the only one beside Tandem which isn't solved in every coord() run. Sometimes
     * coordinated retry is stuck at a local minimum around 2.4. 
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

    double limitVal() {
        return 12.0;
    }

    double stopVal() {
        return stopValFac()*1.9579;
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        Optimizer opt = new DECMA();
        new MessFull().test(100, opt, 50000);
    }

}

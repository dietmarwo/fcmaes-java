package fcmaes.examples;

import java.io.FileNotFoundException;

import fcmaes.core.Log;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Utils;

public class TestGtop {
    
    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Utils.startTiming();
        int numRuns = 100;
        Optimizer opt = new DECMA();
        new Gtoc1().test(numRuns, opt, 10000);
        new Cassini1().test(numRuns, opt, 4000);
        new Cassini2().test(numRuns, opt, 6000);
        new Messenger().test(numRuns, opt, 8000);
        new Rosetta().test(numRuns, opt, 4000);
        new Tandem().test(numRuns, opt, 20000);
        new MessFull().test(numRuns, opt, 50000);
    }

}

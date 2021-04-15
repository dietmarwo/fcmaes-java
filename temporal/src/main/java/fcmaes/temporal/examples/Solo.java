/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.temporal.examples;

import fcmaes.core.Log;
import fcmaes.temporal.core.OptimizerWorker;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solo {

    public static void main(String[] args) throws FileNotFoundException {

        Log.setLog();

        Map<String,String> params = new HashMap<String,String>();

        params.put("fitnessClass", "fcmaes.examples.Solo");
        params.put("optimizerClass", "fcmaes.core.Optimizers$Bite(16)");
        params.put("runs", "20000");
        params.put("maxEvals", "150000");
        params.put("popSize", "31");
        params.put("stopVal", "-1E99");
        params.put("limit", "1E99");

        int numExecs = 1;
        Map<String, List<Double>> xs = OptimizerWorker.runWorkflow(numExecs, params);
    }

}

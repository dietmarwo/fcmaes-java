package fcmaes.temporal.examples;

import fcmaes.temporal.core.SmartWorker;
import fcmaes.core.Log;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MessengerFull {

    public static void main(String[] args) throws FileNotFoundException {

        Log.setLog();
        Map<String,String> params = new HashMap<String,String>();

        params.put("fitnessClass", "fcmaes.examples.MessFull");
        params.put("optimizerClass", "fcmaes.core.Optimizers$DECMA");
        params.put("runs", "20000");
        params.put("startEvals", "1500");
        params.put("popSize", "31");
        params.put("stopVal", "-1E99");
        params.put("limit", "20.0");

        int numExecs = 8;
        List<List<Double>> xs = SmartWorker.runWorkflow(numExecs, params);
    }

}

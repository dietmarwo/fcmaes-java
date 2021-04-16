package fcmaes.temporal.core;

import fcmaes.core.Fitness;
import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.workflow.Async;
import io.temporal.workflow.Promise;
import io.temporal.workflow.Workflow;
import io.temporal.workflow.WorkflowInfo;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Workflow implementation for distributed smart/coordinated parallel optimization retry.
 */
public class SmartWorkflowImpl implements SmartWorkflow {

    // workflow state - only dependent on the signals sent to the workflow
    SmartActivityImpl.Optimize store;
    int signalCount = 0;

    private final SmartActivity smartActivity =
            Workflow.newActivityStub(
                    SmartActivity.class,
                    ActivityOptions.newBuilder()
                            .setScheduleToCloseTimeout(Duration.ofSeconds(20000))
                            .setHeartbeatTimeout(Duration.ofSeconds(20000))
                            .setRetryOptions(
                                    RetryOptions.newBuilder()
                                            .setMaximumAttempts(100)
                                            .setInitialInterval(Duration.ofSeconds(100))
                                            .build())
                            .build());

    //start workflow
    @Override
    public List<List<Double>> optimize(int num, Map<String, String> params) {
        WorkflowInfo wi = Workflow.getInfo();
        try {
            System.out.printf(
                    "\nWorkflow started: WorkflowID: %s RunID: %s, Namespace: %s, Host: %s",
                    wi.getWorkflowId(), wi.getRunId(), wi.getNamespace(), InetAddress.getLocalHost());
        } catch (UnknownHostException e) {
        }
        String fitnessClass = params.get("fitnessClass");
        store = new SmartActivityImpl.Optimize(fitnessClass, new Fitness[500], true);
        List<Promise<String>> promises = new ArrayList<Promise<String>>();
        for (int i = 1; i <= num; i++) {
            System.out.println("optimization " + i + " requested");
            Promise<String> pr = Async.function(smartActivity::optimize, i, params);
            promises.add(pr);
        }
        for (Promise<String> pr : promises) {
            String index = pr.get();
            System.out.println("optimization " + index + " ended");
        }
        return getXs();
    }

    // signal
    @Override
    public void storeFitness(List<Double> ys, List<List<Double>> xs) {
        signalCount++;
        System.out.println("signal " + signalCount + " " + ys.size());
        store.add(ys, xs);
        store.dump(signalCount);
    }

    // query
    @Override
    public List<List<Double>> getFitness(long minTime) {
        List<Double> ys = new ArrayList<Double>();
        List<List<Double>> res = new ArrayList<List<Double>>();
        store.sinceLastTransfer(minTime, ys, res);
        res.add(ys);
        System.out.println("query " + minTime + " " + ys.size());
        return res;
    }

    // query
    @Override
    public List<Double> getYs() {
        return store.getYs();
    }

    // query
    @Override
    public List<List<Double>> getXs() {
        return store.getXs();
    }
}

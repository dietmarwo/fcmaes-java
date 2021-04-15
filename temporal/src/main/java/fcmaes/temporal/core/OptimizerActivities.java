/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.temporal.core;

import fcmaes.core.Fitness;
import fcmaes.core.Optimizers;
import io.temporal.activity.Activity;
import io.temporal.activity.ActivityExecutionContext;
import io.temporal.activity.ActivityInfo;
import io.temporal.activity.ActivityInterface;
import io.temporal.client.WorkflowClient;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.worker.WorkerOptions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * Demonstrates an asynchronous activity implementation. Requires a local instance of Temporal
 * server to be running.
 */
public class OptimizerActivities {

  @ActivityInterface
  public interface OptimizerActivity {
    String optimize(int index, Map<String,String> params);
  }

  static class OptimizerActivityImpl implements OptimizerActivity {


    static long interval = 100000; // ms
    WorkflowClient client;
    OptimizerWorker.OptimizerWorkflow workflow;

    HashMap<String, Double> ymap = new HashMap<String, Double>();
    HashMap<String, List<Double>> xmap = new HashMap<String, List<Double>>();
    HashMap<String, Long> tmap = new HashMap<String, Long>();
    long time = System.currentTimeMillis();

    public OptimizerActivityImpl(WorkflowClient client) {
      this.client = client;
    }

    synchronized private boolean signal(String key, Double y, List<Double> x) {
      if (!ymap.containsKey(key) || y < ymap.get(key)) {
        ymap.put(key, y);
        xmap.put(key, x);
        tmap.put(key, System.currentTimeMillis());
      }
      if (System.currentTimeMillis() - time > interval) {
        int i = 0;
        for (String k: tmap.keySet()) {
          if (tmap.get(k) > time) {
            workflow.optimum(k, ymap.get(k), xmap.get(k));
            i++;
          }
        }
        System.out.println("" + i + " signals sent");
        time = System.currentTimeMillis();
      }
      return false;
    }

    public String optimize(int index, Map<String,String> params) {
      try {

        String fitnessClass = params.get("fitnessClass");
        String optimizerClass = params.get("optimizerClass");
        int runs = Integer.parseInt(params.get("runs"));
        int maxEvals = Integer.parseInt(params.get("maxEvals"));
        int popSize = Integer.parseInt(params.get("popSize"));
        double stopVal = Double.parseDouble(params.get("stopVal"));
        double limit = Double.parseDouble(params.get("limit"));

        Fitness fit = Utils.buildFitness(fitnessClass);
        Optimizers.Optimizer opt = Utils.buildOptimizer(optimizerClass);

        System.out.println("optimize " + index + " " + fitnessClass + " " + optimizerClass);
        ActivityExecutionContext ctx = Activity.getExecutionContext();
        ActivityInfo info = ctx.getInfo();
        System.out.printf(
                "\nActivity started: WorkflowID: %s RunID: %s", info.getWorkflowId(), info.getRunId());
        this.workflow =
                client.newWorkflowStub(OptimizerWorker.OptimizerWorkflow.class, info.getWorkflowId());
        fit._callBack = (key,y,x) -> signal(key,y,x);
        fit.minimizeN(runs, opt, null, maxEvals, stopVal, popSize, limit);
      } catch (Exception ex) {
        ex.printStackTrace();
      }
      return "success";
    }
  }

  public static void main(String[] args) throws ExecutionException, InterruptedException {
    try {
        String url = args.length == 0 ? "127.0.0.1:7233" : args[0];
        WorkflowServiceStubsOptions so = WorkflowServiceStubsOptions.newBuilder()
                .setTarget(url).build();
        // gRPC stubs wrapper that talks to the docker instance of temporal service.
        WorkflowServiceStubs service = WorkflowServiceStubs.newInstance(so);
        // client that can be used to start and signal workflows
        WorkflowClient client = WorkflowClient.newInstance(service);

        // worker factory that can be used to create workers for specific task queues
        WorkerFactory factory = WorkerFactory.newInstance(client);
        // Worker that listens on a task queue and hosts both workflow and activity implementations.

        WorkerOptions wo = WorkerOptions.newBuilder()
                .setMaxConcurrentActivityExecutionSize(1).build();
        Worker worker = factory.newWorker(OptimizerWorker.TASK_QUEUE, wo);

        worker.registerActivitiesImplementations(new OptimizerActivityImpl(client));

        // Start listening to the workflow and activity task queues.
        factory.start();

        Thread.sleep(100000000);
    } catch (Exception ex) {
      ex.printStackTrace();
    }

  }
}

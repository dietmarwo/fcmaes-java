/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.temporal.core;

import fcmaes.core.Fitness;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.RetryOptions;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.workflow.*;
import org.apache.commons.lang.RandomStringUtils;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Implements a smart retry optimization workflow worker. Requires a local instance of Temporal
 * server to be running.
 */
public class SmartWorker {

  static final String TASK_QUEUE = "SmartOptimization";

  /** Workflow interface must have a method annotated with @WorkflowMethod. */
  @WorkflowInterface
  public interface SmartWorkflow {

    @WorkflowMethod
    List<List<Double>> optimize(int num, Map<String,String> params);

    /** Receives new optimum for key. */
    @SignalMethod
    void storeFitness(List<Double> ys, List<List<Double>> xs);

    @QueryMethod
    List<List<Double>> getFitness(long minTime);

    @QueryMethod
    List<Double> getYs();

    @QueryMethod
    List<List<Double>> getXs();
  }

  public static class SmartWorkflowImpl implements SmartWorkflow {

    SmartActivities.Optimize store;
    int signalCount = 0;

    private final SmartActivities.SmartActivity smartActivity =
        Workflow.newActivityStub(
            SmartActivities.SmartActivity.class,
            ActivityOptions.newBuilder()
                .setScheduleToCloseTimeout(Duration.ofSeconds(20000))
                .setHeartbeatTimeout(Duration.ofSeconds(20000))
                .setRetryOptions(
                    RetryOptions.newBuilder()
                        .setMaximumAttempts(100)
                        .setInitialInterval(Duration.ofSeconds(100))
                        .build())
                .build());

    @Override
    public List<List<Double>> optimize(int num, Map<String,String> params) {
      WorkflowInfo wi = Workflow.getInfo();
      try {
        System.out.printf(
                "\nWorkflow started: WorkflowID: %s RunID: %s, Namespace: %s, Host: %s",
                wi.getWorkflowId(), wi.getRunId(), wi.getNamespace(), InetAddress.getLocalHost());
      } catch (UnknownHostException e) {}
      String fitnessClass = params.get("fitnessClass");
      store = new SmartActivities.Optimize(fitnessClass, new Fitness[500], true);
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

    @Override
    public void storeFitness(List<Double> ys, List<List<Double>> xs) {
      signalCount++;
      System.out.println("signal " + signalCount + " " + ys.size());
      store.add(ys, xs);
      store.dump(signalCount);
    }

    @Override
    public List<List<Double>> getFitness(long minTime) {
      List<Double> ys = new ArrayList<Double>();
      List<List<Double>> res = new ArrayList<List<Double>>();
      store.sinceLastTransfer(minTime, ys, res);
      res.add(ys);
      System.out.println("query " + minTime + " " + ys.size());
      return res;
    }

    @Override
    public List<Double> getYs() {
      return store.getYs();
    }

    @Override
    public List<List<Double>> getXs() {
      return store.getXs();
    }
  }

  public static List<List<Double>> runWorkflow(int numExecs, Map<String,String> params)  {
    try {
      // Start a worker that hosts the workflow implementation.
      WorkflowServiceStubs service = WorkflowServiceStubs.newInstance();
      WorkflowClient client = WorkflowClient.newInstance(service);
      WorkerFactory factory = WorkerFactory.newInstance(client);
      Worker worker = factory.newWorker(TASK_QUEUE);
      worker.registerWorkflowImplementationTypes(SmartWorkflowImpl.class);
      factory.start();

      String workflowId = RandomStringUtils.randomAlphabetic(10);

      SmartWorkflow workflow =
          client.newWorkflowStub(
              SmartWorkflow.class,
              WorkflowOptions.newBuilder()
                  .setTaskQueue(TASK_QUEUE)
                  .setWorkflowId(workflowId)
                  .setWorkflowTaskTimeout(Duration.ofSeconds(20000))
                  .setWorkflowExecutionTimeout(Duration.ofSeconds(20000))
                  .setWorkflowRunTimeout(Duration.ofSeconds(20000))
                  .build());

      CompletableFuture<List<List<Double>>> xs =
              WorkflowClient.execute(workflow::optimize, numExecs, params);
      return xs.get();
    } catch (Exception ex) {
      ex.printStackTrace();
      return null;
    }
  }

}

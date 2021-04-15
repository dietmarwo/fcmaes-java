/*
 *  Copyright (c) 2020 Temporal Technologies, Inc. All Rights Reserved
 *
 *  Copyright 2012-2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Modifications copyright (C) 2017 Uber Technologies, Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"). You may not
 *  use this file except in compliance with the License. A copy of the License is
 *  located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 *  or in the "license" file accompanying this file. This file is distributed on
 *  an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied. See the License for the specific language governing
 *  permissions and limitations under the License.
 */

package fcmaes.temporal.core;

import fcmaes.core.Log;
import io.temporal.activity.ActivityOptions;
import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.common.RetryOptions;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.workflow.*;
import org.apache.commons.lang3.RandomStringUtils;

import java.io.FileNotFoundException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 */
public class OptimizerWorker {

  static final String TASK_QUEUE = "FcmaesOptimization";

  /** Workflow interface must have a method annotated with @WorkflowMethod. */
  @WorkflowInterface
  public interface OptimizerWorkflow {

    @WorkflowMethod
    Map<String, List<Double>> optimize(int num, Map<String,String> params);

    /** Receives new optimum for key. */
    @SignalMethod
    void optimum(String key, double y, List<Double> x);

    @QueryMethod
    Map<String, Double> getYMap();

    @QueryMethod
    Map<String, List<Double>> getXMap();
  }

  public static class OptimizerWorkflowImpl implements OptimizerWorkflow {

    ConcurrentHashMap<String, Double> ymap = new ConcurrentHashMap<String, Double>();
    ConcurrentHashMap<String, List<Double>> xmap = new ConcurrentHashMap<String, List<Double>>();
    ConcurrentHashMap<String, Long> tmap = new ConcurrentHashMap<String, Long>();

    private final OptimizerActivities.OptimizerActivity optimizer =
        Workflow.newActivityStub(
            OptimizerActivities.OptimizerActivity.class,
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
    public Map<String, List<Double>> optimize(int num, Map<String,String> params) {
      WorkflowInfo wi = Workflow.getInfo();
      try {
        System.out.printf(
                "\nWorkflow started: WorkflowID: %s RunID: %s, Namespace: %s, Host: %s",
                wi.getWorkflowId(), wi.getRunId(), wi.getNamespace(), InetAddress.getLocalHost());
      } catch (UnknownHostException e) {}

      List<Promise<String>> promises = new ArrayList<Promise<String>>();
      for (int i = 1; i <= num; i++) {
        System.out.println("optimization " + i + " requested");
        Promise<String> pr = Async.function(optimizer::optimize, i, params);
        promises.add(pr);
      }
      for (Promise<String> pr : promises) {
        String index = pr.get();
        System.out.println("optimization " + index + " ended");
      }
      return xmap;
    }

    @Override
    public void optimum(String key, double y, List<Double> x) {
      if (!ymap.containsKey(key) || y < ymap.get(key)) {
        ymap.put(key, y);
        xmap.put(key, x);
        tmap.put(key, System.currentTimeMillis());
        System.out.println("signal " + key + " " + y + " " + Arrays.toString(x.toArray()));
      }
    }

    @Override
    public Map<String, Double> getYMap() {
      return ymap;
    }

    @Override
    public Map<String, List<Double>> getXMap() {
       return xmap;
    }
  }

  public static Map<String, List<Double>> runWorkflow(int numExecs, Map<String,String> params) {
    try {
      // Start a worker that hosts the workflow implementation.
      WorkflowServiceStubs service = WorkflowServiceStubs.newInstance();
      WorkflowClient client = WorkflowClient.newInstance(service);
      WorkerFactory factory = WorkerFactory.newInstance(client);
      Worker worker = factory.newWorker(TASK_QUEUE);
      worker.registerWorkflowImplementationTypes(OptimizerWorkflowImpl.class);
      factory.start();

      String workflowId = RandomStringUtils.randomAlphabetic(10);

      OptimizerWorker.OptimizerWorkflow workflow =
              client.newWorkflowStub(
                      OptimizerWorker.OptimizerWorkflow.class,
                      WorkflowOptions.newBuilder()
                              .setTaskQueue(TASK_QUEUE)
                              .setWorkflowId(workflowId)
                              .setWorkflowTaskTimeout(Duration.ofSeconds(20000))
                              .setWorkflowExecutionTimeout(Duration.ofSeconds(20000))
                              .setWorkflowRunTimeout(Duration.ofSeconds(20000))
                              .build());

      CompletableFuture<Map<String, List<Double>>> xmap =
              WorkflowClient.execute(workflow::optimize, numExecs, params);
      return xmap.get();
    } catch (Exception ex) {
      ex.printStackTrace();
      return null;
    }
  }

}

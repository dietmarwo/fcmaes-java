/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.temporal.core;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import org.apache.commons.lang.RandomStringUtils;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Implements a distributed smart/coordinated parallel optimization retry workflow worker.
 * Requires a local instance of Temporal server to be running.
 */
public class SmartRetryWorker {

  static final String TASK_QUEUE = "SmartOptimization";

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

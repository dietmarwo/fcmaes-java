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
import org.apache.commons.lang3.RandomStringUtils;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Implements an distributed parallel optimization retry workflow worker.
 * Requires a local instance of Temporal server to be running.
 */
public class OptimizerRetryWorker {

  static final String TASK_QUEUE = "FcmaesOptimization";

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

      OptimizerWorkflow workflow =
              client.newWorkflowStub(
                      OptimizerWorkflow.class,
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

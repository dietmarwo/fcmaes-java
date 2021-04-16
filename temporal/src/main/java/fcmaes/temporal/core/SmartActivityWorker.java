/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.temporal.core;

import fcmaes.core.Fitness;
import fcmaes.core.Optimizers.Optimizer;
import io.temporal.client.WorkflowClient;
import io.temporal.serviceclient.WorkflowServiceStubs;
import io.temporal.serviceclient.WorkflowServiceStubsOptions;
import io.temporal.worker.Worker;
import io.temporal.worker.WorkerFactory;
import io.temporal.worker.WorkerOptions;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * Implements a distributed smart/coordinated parallel optimization retry activity worker.
 * Requires a local instance of Temporal server to be running.
 * Receives the target URL of the temporal server as argument.
 */
public class SmartActivityWorker {

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
            // Worker that listens on a task queue and hosts the activity implementation.
            // Only one activity is running. Parallelism is implemented by the smart/coordinated retry.
            WorkerOptions wo = WorkerOptions.newBuilder()
                    .setMaxConcurrentActivityExecutionSize(1).build();
            Worker worker = factory.newWorker(SmartRetryWorker.TASK_QUEUE, wo);
            worker.registerActivitiesImplementations(new SmartActivityImpl(client));
            // Start listening to the activity task queue.
            factory.start();
            // keep main thread alive - to be killed manually
            Thread.sleep(100000000);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}

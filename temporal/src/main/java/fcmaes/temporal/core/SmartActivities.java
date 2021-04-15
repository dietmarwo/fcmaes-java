/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.temporal.core;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Optimizers;
import fcmaes.core.Optimizers.Optimizer;
import fcmaes.core.Threads;
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
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class SmartActivities {

    @ActivityInterface
    public interface SmartActivity {
        String optimize(int index, Map<String,String> params);
    }

    static class SmartActivityImpl extends CoordRetry implements SmartActivity {

        WorkflowClient client;
        SmartWorker.SmartWorkflow workflow;

        SmartActivityImpl(WorkflowClient client) {
            this.client = client;
        }

        public String optimize(int index, Map<String,String> params) {
            try {
                String fitnessClass = params.get("fitnessClass");
                String optimizerClass = params.get("optimizerClass");
                int runs = Integer.parseInt(params.get("runs"));
                int startEvals = Integer.parseInt(params.get("startEvals"));
                int popSize = Integer.parseInt(params.get("popSize"));
                double stopVal = Double.parseDouble(params.get("stopVal"));
                double limitVal = Double.parseDouble(params.get("limit"));
                boolean log = false;

                Fitness fit = Utils.buildFitness(fitnessClass);
                Optimizers.Optimizer opt = Utils.buildOptimizer(optimizerClass);

                System.out.println("optimize " + index + " " + fitnessClass + " " + optimizerClass);
                ActivityExecutionContext ctx = Activity.getExecutionContext();
                ActivityInfo info = ctx.getInfo();
                System.out.printf(
                        "\nActivity started: WorkflowID: %s RunID: %s", info.getWorkflowId(), info.getRunId());

                workflow = client.newWorkflowStub(SmartWorker.SmartWorkflow.class, info.getWorkflowId());
                Fitness[] store = new Fitness[500];
                SmartActivities.Optimize retry = new SmartActivities.Optimize(
                        runs, fit, null, store, limitVal, stopVal,
                        startEvals, 0, 0, opt, 0, log, workflow);
                Threads threads = new Threads(retry, 8);
                threads.start();
                threads.join();
                retry.sort();
                retry.dump(retry.countRuns());
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            return "success";
        }
     }

     public static class Optimize extends fcmaes.core.CoordRetry.Optimize {

        long _lastTransferTime = 0;
        SmartWorker.SmartWorkflow workflow;

        public Optimize(int runs, Fitness fit, double[] guess, Fitness[] store, double limitVal,
                        double stopVal, int startEvals,
                        int maxEvalFac, int checkInterval, Optimizer opt, int popsize, boolean log,
                        SmartWorker.SmartWorkflow workflow) {
            super(runs, fit, guess, store, limitVal, stopVal, startEvals,
                    maxEvalFac, checkInterval, opt, popsize, log);
            this.workflow = workflow;
        }

        public Optimize(String fitnessClass, Fitness[] store, boolean log) {
            super(Utils.buildFitness(fitnessClass), store, log);
        }

        void sinceLastTransfer(long minTime, List<Double> ys, List<List<Double>> xs) {
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = (Fitness) _store[i];
                if (fit._time > minTime) {
                    ys.add(fit._bestY);
                    xs.add(Arrays.asList(ArrayUtils.toObject(fit._bestX)));
                }
            }
        }

        List<Double> getYs() {
            List<Double> ys = new ArrayList<Double>();
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = (Fitness) _store[i];
                ys.add(fit._bestY);
            }
            return ys;
        }

        List<List<Double>> getXs() {
            List<List<Double>> xs = new ArrayList<List<Double>>();
            for (int i = 0; i < _numStored; i++) {
                Fitness fit = (Fitness) _store[i];
                xs.add(Arrays.asList(ArrayUtils.toObject(fit._bestX)));
            }
            return xs;
        }

        public void add(List<Double> ys, List<List<Double>> xs) {
            for (int i = 0; i < ys.size(); i++) {
                double[] x = ArrayUtils.toPrimitive(xs.get(i).toArray(Double[]::new));
                Fitness fit = new Fitness(x.length);
                fit._bestX = x;
                fit._bestY = ys.get(i);
                addResult(countRuns(), fit, Double.MAX_VALUE);
            }
        }

        @Override
        public void sort() {
            super.sort();
            long now = System.currentTimeMillis();
            if (workflow != null && _lastTransferTime + 10000 < now) {
                //communicate with temporal server
                List<Double> ys = new ArrayList<Double>();
                List<List<Double>> xs = new ArrayList<List<Double>>();
                sinceLastTransfer(_lastTransferTime, ys, xs);
                List<List<Double>> xsOthers = workflow.getFitness(_lastTransferTime);
                System.out.println("query " + (now - _lastTransferTime) + " " + xsOthers.size());
                workflow.storeFitness(ys, xs);
                System.out.println("signal " + ys.size());
                _lastTransferTime = now;
                int last = xsOthers.size() - 1;
                if (last > 0)
                    add(xsOthers.get(last), xsOthers.subList(0, last));
            }
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

            //int threadNum = Math.min(16, Runtime.getRuntime().availableProcessors());
            WorkerOptions wo = WorkerOptions.newBuilder()
                    .setMaxConcurrentActivityExecutionSize(4).build();
            Worker worker = factory.newWorker(SmartWorker.TASK_QUEUE, wo);

            worker.registerActivitiesImplementations(new SmartActivities.SmartActivityImpl(client));
            // Start listening to the workflow and activity task queues.
            factory.start();
            Thread.sleep(100000000);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}

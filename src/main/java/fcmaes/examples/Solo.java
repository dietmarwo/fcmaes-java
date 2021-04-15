package fcmaes.examples;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.lang3.ArrayUtils;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.Vector3D;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Jni;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.Bite;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.Result;
import fcmaes.core.Utils;
import fcmaes.kepler.Kepler;
import fcmaes.kepler.RVT;
import fcmaes.kepler.Resonance;

public class Solo extends Fitness {

    /*
     * Works not on Windows! Use the Linux subsystem for Windows there.
     * 
     * This code is derived from https://github.com/esa/pykep/pull/127 originally
     * developed by Moritz v. Looz @mlooz . It was modified following suggestions
     * from Waldemar Martens @MartensWaldemar_gitlab In this implementation there
     * are restrictions regarding the allowed planet sequence which will be removed
     * in later revisions. The code is designed around an "orbit abstraction" class
     * RVT simplifying the definition of the objective function. This problem is
     * quite a challenge for state of the art optimizers, but good solutions
     * fulfilling the requirements can be found. See
     * https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter
     * 
     * This Java code is about factor 3.6 faster than the equivalent Python code
     * https://github.com/dietmarwo/fast-cma-es/blob/master/examples/
     * solar_orbiter_udp.py
     */

    static final int[][][] resos0_ = new int[][][] { { { 1, 1 }, { 5, 4 }, { 4, 3 } }, { { 1, 1 }, { 5, 4 }, { 4, 3 } },
        { { 1, 1 }, { 5, 4 }, { 4, 3 } }, { { 4, 3 }, { 3, 2 }, { 5, 3 } }, { { 4, 3 }, { 3, 2 }, { 5, 3 } },
        { { 4, 3 }, { 3, 2 }, { 5, 3 } } };

    static int[][][] resos_ = resos0_;

    static String resoFile_ = "i3resos4.txt";

    static final int earth = 3;

    static final int venus = 2;

    static final int max_revolutions = 2;

    static final double maxLaunchDV = 5600;

    static final double safe_distance = 350000;

    static final double min_dist_sun = 0.28; // AU

    static final double max_dist_sun = 1.2; // AU

    static final double max_mission_time = 11.0 * Utils.YEAR;

    static double max_log_y_value = 12000;

    static final long log_interval_evals = 200000000;
//    static final long log_interval_evals = 10000000;

    static final double AU = 1.49597870691e11; // m

    static final double theta = Math.toRadians(7.25);

    Vector3D _rotation_axis;

    int[][] _usedResos = null;

    List<Double> logData = null;

    public Solo() {
        super(10);
        init();
    }

    Vector3D rotate_vector(Vector3D v) {
        Rotation rot = new Rotation(_rotation_axis, theta, RotationConvention.FRAME_TRANSFORM);
        return rot.applyTo(v);
    }

    void init() {
        double t_plane_crossing = 7645;
        double[] r = new double[3];
        double[] v = new double[3];
        Jni.planetEplC(3, t_plane_crossing, r, v);
        _rotation_axis = Utils.vector(r).normalize();
    }

    public Solo create() {
        Solo solo = new Solo();
        solo._rotation_axis = _rotation_axis;
        solo._callBack = _callBack;
        return solo;
    }

    double period(int pli, double time) {
        double[] data = new double[5];
        Jni.planetDataC(pli, time / Utils.DAY, data);
        return data[0];
    }

    public double[] guess() {
        return Utils.rnd(lower(), upper());
    }

    public double[] lower() {
        return new double[] { 7000, 50, 50, 50, -Math.PI, -Math.PI, -Math.PI, -Math.PI, -Math.PI, -Math.PI };
    }

    public double[] upper() {
        return new double[] { 8000, 420, 400, 400, Math.PI, Math.PI, Math.PI, Math.PI, Math.PI, Math.PI };
    }

    public String toString() {
        StringBuffer sb = new StringBuffer("c= " + _evals + " v= " + Utils.r(_bestY, 10));
        sb.append(" ");
        sb.append(Arrays.toString(_bestX));
        return sb.toString();
    }

    static Vector3D v_planet(int pli, double time) {
        double[] r = new double[3];
        double[] v = new double[3];
        Jni.planetEplC(pli, time / Utils.DAY, r, v);
        return Utils.vector(v);
    }

    static double ga_dv(int pli, double time, Vector3D vin, Vector3D vout) {
        Vector3D vpl = v_planet(pli, time);
        double[] v_rel_in = Utils.array(vin.subtract(vpl));
        double[] v_rel_out = Utils.array(vout.subtract(vpl));
        return Jni.fb_vel(v_rel_in, v_rel_out, pli);
    }

    static void mga(RVT in, int pli1, int pli2, double time, double tof, Vector3D[] vout_in, List<RVT> outs,
            List<Double> dvs) {
        RVT planet2 = new RVT(pli2, (time + tof) / Utils.DAY);
        Vector3D[] lambert = in.bestLambert(planet2, pli1, false, max_revolutions);
        vout_in[0] = lambert[0];
        vout_in[1] = lambert[1];
        RVT out = new RVT(in);
        out.setV(lambert[0]);
        outs.add(out);
        dvs.add(ga_dv(pli1, time, in.v(), lambert[0]));
    }

    RVT rotate(int pli, double time, Vector3D vin) {
        RVT rvt = new RVT(pli, time / Utils.DAY);
        rvt.setR(rotate_vector(rvt.r()));
        rvt.setV(rotate_vector(vin));
        return rvt;
    }

    static RVT rvt(int pli, double time, Vector3D vout) {
        RVT rvt = new RVT(pli, time / Utils.DAY);
        rvt.setV(vout);
        return rvt;
    }

    private static class Trajectory {
        double y;
        double[] x;
        long hash;

        Trajectory(double y, double[] x, long hash) {
            this.y = y;
            this.x = x;
            this.hash = hash;
        }

        String resos() {
            Solo solo = new Solo();
            solo.logData = new ArrayList<Double>();
            solo.eval(x);
            return "" + y + " " + Utils.r(solo._usedResos, "") + " " + Arrays.toString(solo.logData.toArray()) + " "
                    + Arrays.toString(x);
        }

        int[][][] used_resos() {
            Solo solo = new Solo();
            solo.logData = new ArrayList<Double>();
            solo.eval(x);
            int[][][] resos = new int[6][1][];
            for (int i = 0; i < 6; i++)
                resos[i][0] = solo._usedResos[i];
            return resos;
        }
        
        static int count = 1;

        String pretty() {
            Solo solo = new Solo();
            solo.logData = new ArrayList<Double>();
            solo.eval(x);
            StringBuffer buf = new StringBuffer();
            buf.append("|" + (count++) + " |" + Utils.r(y) + " |" + Utils.r(solo._usedResos, ""));
            for (double v : solo.logData)
                buf.append(" |" + Utils.r(v));
            return buf.toString();
        }

        String prettyDetails() {
            Solo solo = new Solo();
            solo.logData = new ArrayList<Double>();
            solo.eval(x);
            StringBuffer buf = new StringBuffer();
            buf.append("|" + (count++) + " |" + Utils.r(y) + " |" + Utils.r(solo._usedResos, ""));
            buf.append(" |" + Arrays.toString(x));
            return buf.toString();
        }
    }

    static double bestY_ = 1E99;
    static ConcurrentHashMap<Long, Trajectory> bestTrajectories_ = new ConcurrentHashMap<Long, Trajectory>();
    static AtomicLong counter_ = new AtomicLong(0);
    static List<RVT> outs_; // trajectory orbit log
    static int[][] bestResos_ = new int[6][2]; // used resonances

    void dumpTrajectories(String fname) {
        TreeMap<Double, Trajectory> sorted = new TreeMap<Double, Trajectory>();
        for (Trajectory tra : bestTrajectories_.values())
            sorted.put(tra.y, tra);
        StringBuffer buf = new StringBuffer();
        buf.append("best = " + bestY_ + " evals = " + counter_.get() + " " + Utils.measuredSeconds() + " sec\n");
        for (double y : sorted.keySet()) {
//            if (y < max_log_y_value)
                buf.append(sorted.get(y).resos() + "\n");
        }
        buf.append("\n");
        try {
            Files.write(Paths.get(fname), buf.toString().getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("Cannot write to " + resoFile_ + ": " + e.getMessage());
        }
    }

    void showTrajectories() {
        TreeMap<Double, Trajectory> sorted = new TreeMap<Double, Trajectory>();
        for (Trajectory tra : bestTrajectories_.values())
//            sorted.put(tra.y, tra);
            System.out.println("" + tra.y + " " + tra.resos());
        for (double y : sorted.keySet())
            System.out.println(sorted.get(y).resos());
    }

    void prettyTrajectories(String fname) {
        TreeMap<Double, Trajectory> sorted = new TreeMap<Double, Trajectory>();
        for (Trajectory tra : bestTrajectories_.values())
            sorted.put(tra.y, tra);
        StringBuffer buf = new StringBuffer();
        for (double y : sorted.keySet()) {
//            buf.append(sorted.get(y).prettyDetails() + "\n");
            buf.append(sorted.get(y).pretty() + "\n");
        }
        buf.append("\n");
        try {
            Files.write(Paths.get(fname), buf.toString().getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.err.println("Cannot write to " + resoFile_ + ": " + e.getMessage());
        }
    }

    void improveTrajectories() {
        TreeMap<Double, Trajectory> sorted = new TreeMap<Double, Trajectory>();
        for (Trajectory tra : bestTrajectories_.values())
            sorted.put(tra.y, tra);
        bestTrajectories_ = new ConcurrentHashMap<Long, Trajectory>(); // reset map
        int i = 0;
        for (double y : sorted.keySet()) {
            if (i++ < -1) continue;
            Trajectory tra = sorted.get(y);
            String resstr = tra.resos();
            resos_ = tra.used_resos();
            bestY_ = 1E99;
            Solo fit = create();
            Utils.startTiming();
            
            System.out.println("improve smart DECMA " + i + " " + resstr);
            Result res = CoordRetry.optimize(4000, fit, new DECMA(), null, 100000, 0, 3000, true);
             
            showTrajectories();
            dumpTrajectories(resoFile_);
        }
    }
    
    long hash(int[][] reso) {
        long hash = 0;
        for (int i = 0; i < reso.length; i++) {
            int[] r = reso[i];
            for (int j = 0; j < r.length; j++)
                hash = 31*hash + r[j];
        }
        return hash;
    }

    @Override
    public double eval(double[] x) {
        List<Double> dvs = new ArrayList<Double>();
        double reso_penalty = 0;
        double t;
        double t0 = t = x[0] * Utils.DAY;
        double tof01 = x[1] * Utils.DAY;
        double tof23 = x[2] * Utils.DAY;
        double tof34 = x[3] * Utils.DAY;
        double[] beta = Arrays.copyOfRange(x, 4, 10);
        List<RVT> outs = new ArrayList<RVT>(); // trajectory orbits
        RVT start = new RVT(3, t / Utils.DAY);
        Vector3D[] vout_in = new Vector3D[2];
        mga(start, earth, venus, t, tof01, vout_in, outs, dvs);
        double dvStart = dvs.get(0);
        dvs.set(0, Math.max(0, dvStart - maxLaunchDV));
        int[][][] allres = logData == null ? resos_ : resos0_; 
        int[][] resos = new int[6][];

        t += tof01;
        Resonance res1 = Resonance.resonance(venus, t, vout_in[1], allres[0], beta[0], safe_distance, outs, dvs);
        reso_penalty += res1._dt;
        resos[0] = res1.selected();

        t += res1.tof();
        RVT in = rvt(venus, t, res1._vout);
        mga(in, venus, earth, t, tof23, vout_in, outs, dvs);

        t += tof23;
        in = rvt(earth, t, vout_in[1]);
        mga(in, earth, venus, t, tof34, vout_in, outs, dvs);

        t += tof34;
        Resonance res = null;
        for (int r = 1; r < resos.length; r++) {
            Vector3D v_in = res != null ? res._vout : vout_in[1];
            res = Resonance.resonance(venus, t, v_in, allres[r], beta[r], safe_distance, outs, dvs);
            reso_penalty += res._dt;
            resos[r] = res.selected();
            t += res.tof();
        }
        RVT fin = rotate(venus, t, res._vout);
        Kepler finKep = fin.kepler();

        // orbit should be as polar as possible, but we do not care about
        // prograde/retrograde
        double corrected_inclination = Math.toDegrees(Math.abs(Math.abs(finKep.i() % Math.PI - Math.PI / 2)));
        double emp_perhelion = 2;
        double min_sun_distance = 2;
        double max_sun_distance = 0;

        for (int i = 0; i < outs.size(); i++) {
            RVT rvt0 = outs.get(i);
            double t1 = i < outs.size() - 1 ? outs.get(i + 1).t() : t;
            double dt = t1 - rvt0.t();
            Kepler kep = rvt0.kepler();
            double period = kep.period(rvt0.mu());
            double perhelion = kep.periapsis() / AU;
            min_sun_distance = Math.min(min_sun_distance, perhelion);
            if (i >= outs.size() - 3)
                emp_perhelion = Math.min(emp_perhelion, perhelion);
            if (dt > period)
                max_sun_distance = Math.max(max_sun_distance, kep.apoapsis() / AU);
        }
        double distance_penalty = Math.max(0, min_dist_sun - min_sun_distance);
        distance_penalty += Math.max(0, max_sun_distance - max_dist_sun);

        if (logData != null) {
            logData.add(Math.toDegrees(finKep.i()));
            logData.add(min_sun_distance);
            logData.add(max_sun_distance);
            logData.add(emp_perhelion);
            logData.add((t - t0) / Utils.DAY / Utils.YEAR);
            logData.add(Utils.sum(dvs));
        }

        if (min_sun_distance < min_dist_sun)
            min_sun_distance += 10 * (min_dist_sun - min_sun_distance);

        double time_val = (t - t0) / Utils.DAY;
        if (time_val > max_mission_time)
            time_val += 10 * (time_val - max_mission_time);

        double value = 100 * Utils.sum(dvs) + reso_penalty + 100 * (corrected_inclination)
                + 5000 * (Math.max(0, min_sun_distance - min_dist_sun))
                + 5000 * (Math.max(0, emp_perhelion - min_dist_sun)) + 0.5 * time_val + 50000 * distance_penalty;
        _usedResos = resos;

        // store best trajectory for a given resonance sequence
        if (logData == null) {
            long hash = hash(resos);
            if (!bestTrajectories_.containsKey(hash) || value < bestTrajectories_.get(hash).y) {
                bestTrajectories_.put(hash, new Trajectory(value, x, hash));
                if (value < max_log_y_value)
                	callBack(Utils.r(resos, ""), value, x);
            }
            if (counter_.getAndIncrement() % log_interval_evals == log_interval_evals - 1)
                dumpTrajectories(resoFile_);
        }
        if (value < bestY_ && value < max_log_y_value) {
            bestY_ = value;
            bestResos_ = resos;
            outs_ = outs;
            if (logData == null)
                System.out.println("" + Utils.r(Math.toDegrees(finKep.i())) + " " + Utils.r(min_sun_distance) + " "
                    + Utils.r(max_sun_distance) + " " + Utils.r(emp_perhelion) + " " + Utils.r(Utils.sum(dvs)) + " "
                    + Utils.r(reso_penalty) + " " + Utils.r(time_val) + " " + Utils.r(resos, "") + " " + Utils.r(value)
                    + " " + Arrays.toString(x));
        }
        return value;
    }
	
    void check_good_solutions() {
        double[][] xs = new double[][] { { 7456.679026533558, 399.69805305303163, 164.18980290024066, 334.2488325479292,
                0.1630833705342643, -0.22564954170853885, 0.9805628198484782, 0.4977489581063494, 0.5363419684307973,
                -0.3461410243531474 }, };
        for (double[] x : xs) {
            double y = eval(x);
            Trajectory tra = new Trajectory(y, x, 0);
            System.out.println(tra.resos());
        }
    }

    ConcurrentHashMap<Long, Trajectory> load_logs(String dirName, String pattern, String fname, boolean reeval) {
    	ConcurrentHashMap<Long, Trajectory> trajectories = new ConcurrentHashMap<Long, Trajectory>();
        List<String> lines = Utils.readFiles(dirName, pattern);
        for (String line : lines) {
            String[] tokens = line.split(" ");
            if (tokens.length < 12 || "".equals(tokens[0]) || "Result".equals(tokens[0]))
                continue;
            double y = Double.parseDouble(tokens[0]);
            String args = line.substring(line.lastIndexOf("[") + 1, line.length() - 1);
            double[] x = ArrayUtils
                    .toPrimitive(Arrays.stream(args.split(", ")).map(Double::parseDouble).toArray(Double[]::new));
            long hash = tokens[1].hashCode();
            if (reeval) {
                Solo solo = new Solo();
                solo.logData = new ArrayList<Double>();
                y = solo.eval(x);
                hash = hash(solo._usedResos);
            }
            
            if (!trajectories.containsKey(hash) || y < trajectories.get(hash).y)
            	trajectories.put(hash, new Trajectory(y, x, hash));
        }
        return trajectories;
    }

    Map<String,Map<Long, Trajectory>> load_algo_logs(String dirName, String pattern, String fname, boolean reeval) {
    	Map<String,Map<Long, Trajectory>> trajectories = new HashMap<String,Map<Long, Trajectory>>();
    			
        List<String> lines = Utils.readFiles(dirName, pattern);
        String algo = null;
        Map<Long, Trajectory> algomap = null;
        for (String line : lines) {
            String[] tokens = line.split(" ");
            if ("algo".equals(tokens[0])) {
            	algo = tokens[1];
            	algomap = new HashMap<Long, Trajectory>();
            	trajectories.put(algo, algomap);
            }
            if (tokens.length < 12 || "".equals(tokens[0]) || "Result".equals(tokens[0]))
                continue;
            double y = Double.parseDouble(tokens[0]);
            String args = line.substring(line.lastIndexOf("[") + 1, line.length() - 1);
            double[] x = ArrayUtils
                    .toPrimitive(Arrays.stream(args.split(", ")).map(Double::parseDouble).toArray(Double[]::new));
            long hash = tokens[1].hashCode();
            if (reeval) {
                Solo solo = new Solo();
                solo.logData = new ArrayList<Double>();
                y = solo.eval(x);
                hash = hash(solo._usedResos);
            }
            if (!algomap.containsKey(hash) || y < algomap.get(hash).y)
            	algomap.put(hash, new Trajectory(y, x, hash));
        }
        return trajectories;
    }

    void join_logs(String dirName, String pattern, String fname, boolean reeval) {
        bestTrajectories_ = load_logs(dirName, pattern, fname, reeval);
//        prettyTrajectories(fname);
        dumpTrajectories(fname);
//        improveTrajectories();
    }

    Map<String,List<Double>> eval_logs() {
    	Map<Long, Trajectory> best = load_logs("logs", "solo_results", "collected.txt", true);
        TreeMap<Double, Trajectory> sorted = new TreeMap<Double, Trajectory>();
        Map<Long, Double> sortedkey = new HashMap<Long, Double>();
        for (Entry<Long,Trajectory> e : best.entrySet()) {
        	Trajectory tra = e.getValue();
        	sorted.put(tra.y, tra);
        }
        StringBuffer buf = new StringBuffer();
        int i = 0;
        for (double y : sorted.keySet()) {
        	Trajectory tra = sorted.get(y);
            buf.append(tra.pretty() + "\n");
        	sortedkey.put(tra.hash, tra.y);
        	if (i++ > 100)
        		break;
        }
        buf.append("\n");
        
        Map<String,Map<Long, Trajectory>> algomaps = load_algo_logs("logs", "singleresos1", "collected.txt", true);
        Map<String,List<Double>> algodiffs = new HashMap<String,List<Double>>();
        for (String algo: algomaps.keySet()) {
        	Map<Long, Trajectory> algomap = algomaps.get(algo);
        	List<Double> diffs = new ArrayList<Double>();
        	algodiffs.put(algo, diffs);

            TreeMap<Double, Trajectory> algosorted = new TreeMap<Double, Trajectory>();
        	Map<Double, Long> algokey = new HashMap<Double, Long>();
            for (Entry<Long,Trajectory> e : algomap.entrySet()) {
            	Long key = e.getKey();
             	Trajectory tra = e.getValue();
               	if (!sortedkey.containsKey(tra.hash))
            		continue;
            	Trajectory btra  = best.get(key);
            	double diff = tra.y - btra.y;
            	algosorted.put(diff, tra);
            	algokey.put(diff, e.getKey());
            }
            for (double d : algosorted.keySet()) 
            	diffs.add(d);            
            double[] df = ArrayUtils.toPrimitive(diffs.toArray(Double[]::new));
        	buf.append("['" + algo + "', " + Utils.r(df,1) + "],\n");
        }
        System.out.println(buf.toString());
        return algodiffs;
    }
    
    Solo optimize() {
        Utils.startTiming();
        Solo fit = create();
        fit.minimizeN(12800000, new Bite(16), 120000, 0, 31, 1E99);
//        CoordRetry.optimize(80000, this, new DE(), null, 1E99, 0, 2000, true);
//        CoordRetry.optimize(80000, this, new DECMA(), null, 1E99, 0, 2000, true);
//        CoordRetry.optimize(80000, this, new DeBite(), null, 1E99, 0, 2000, true);        
        System.out.println(fit);
        System.out.println(Utils.measuredMillis() + " ms");
        return fit;
    }


    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Solo solo = new Solo();
        solo.optimize();
//        solo.check_good_solutions();
//        solo.join_logs("logs", "resos", "collected.txt", true);
//        solo.eval_logs();
    }

}

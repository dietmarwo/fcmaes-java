package fcmaes.examples;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.Vector3D;

import fcmaes.core.CoordRetry;
import fcmaes.core.Fitness;
import fcmaes.core.Jni;
import fcmaes.core.Log;
import fcmaes.core.Optimizers.Bite;
import fcmaes.core.Optimizers.DE;
import fcmaes.core.Optimizers.DECMA;
import fcmaes.core.Optimizers.DeBite;
import fcmaes.core.Utils;
import fcmaes.kepler.Kepler;
import fcmaes.kepler.RVT;
import fcmaes.kepler.Resonance;

public class Solo extends Fitness {
    
    /*
     * Works not on Windows! Use the Linux subsystem for Windows there.
     * 
     * This code is derived from https://github.com/esa/pykep/pull/127 
     * originally developed by Moritz v. Looz @mlooz . 
     * It was modified following suggestions from Waldemar Martens @MartensWaldemar_gitlab
     * In this implementation there are restrictions regarding the allowed planet 
     * sequence which will be removed in later revisions.
     * The code is designed around an "orbit abstraction" class RVT simplifying the
     * definition of the objective function. 
     * This problem is quite a challenge for state of the art optimizers, but
     * good solutions fulfilling the requirements can be found.
     * See https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter
     * 
     * This Java code is about factor 3.3 faster than the equivalent Python code
     * https://github.com/dietmarwo/fast-cma-es/blob/master/examples/solar_orbiter_udp.py
     */
    
    static final int[][][] resos_ = new int[][][] {
        {{1,1}, {4,5}, {3,4}},
        {{1,1}, {4,5}, {3,4}},
        {{4,5}, {3,4}, {2,3}, {3,5}},
        {{3,4}, {2,3}, {3,5}},
        {{3,4}, {2,3}, {3,5}},
        {{2,3}, {3,5}}
        };
    
    static final double maxLaunchDV = 5600;
    
    static final double safe_distance = 350000;

    static final double min_dist_sun = 0.28; // AU

    static final double max_dist_sun = 1.2; // AU

    static final double max_mission_time = 11.6*Utils.YEAR;

    static final double AU = 1.49597870691e11; // m
    
    int[][] _resos = new int[6][2];
    
    Vector3D _rotation_axis;
    double _theta = Math.toRadians(7.25);
    List<RVT> _outs; // trajectory orbit log 
     
    Solo() {
        super(11);
        init();
    }

    Vector3D rotate_vector(Vector3D v) {
        Rotation rot = new Rotation(_rotation_axis, _theta, RotationConvention.VECTOR_OPERATOR);
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
        return solo;
    }

    double period(int pli, double time) {
        double[] data = new double[5];
        Jni.planetDataC(pli, time/Utils.DAY, data);
        return data[0];
    }
 
    public double[] guess() {
        return Utils.rnd(lower(), upper());
    }

    public double[] lower() {
        return new double[] {7000, 50, 50, 50, -Math.PI, -Math.PI, -Math.PI, -Math.PI, -Math.PI, -Math.PI, -Math.PI};
    }

    public double[] upper() {
        return new double[] {8000, 400, 400, 400, Math.PI, Math.PI, Math.PI, Math.PI, Math.PI, Math.PI, Math.PI};
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
        Jni.planetEplC(pli, time/Utils.DAY, r, v);
        return Utils.vector(v);
    }
    
    static double ga_dv(int pli, double time, Vector3D vin, Vector3D vout) {
        Vector3D vpl = v_planet(pli, time);
        double[] v_rel_in = Utils.array(vin.subtract(vpl));
        double[] v_rel_out = Utils.array(vout.subtract(vpl));
        return Jni.fb_vel(v_rel_in, v_rel_out, pli);
    }
 
    static double mga(RVT in, int pli1, int pli2, double time, double tof, Vector3D[] vout_in, List<RVT> outs) {
        RVT planet2 = new RVT(pli2, (time+tof)/Utils.DAY);
        Vector3D[] lambert = in.bestLambert(planet2, pli1, false, 2);
        vout_in[0] = lambert[0];
        vout_in[1] = lambert[1];
        RVT out = new RVT(in);
        out.setV(lambert[0]);
        outs.add(out);
        return ga_dv(pli1, time, in.v(), lambert[0]);
    }

    RVT fb_prop_rotate(int pli, double time, Vector3D vin, double beta) {
        Vector3D vpl = v_planet(pli, time);
        double[] data = new double[5];
        Jni.planetDataC(pli, time/Utils.DAY, data);
        double mu = data[2];
        double rp = data[3] + safe_distance;
        double[] vout = new double[3];
        Jni.fb_prop(Utils.array(vin), Utils.array(vpl), rp, beta, mu, vout);
        Vector3D r = new RVT(2, time/Utils.DAY).r();
        r = rotate_vector(r);
        Vector3D v = rotate_vector(Utils.vector(vout));
        RVT fin = rvt(2, time, v);
        fin.setR(r);
        return fin;
    }
   
    static RVT rvt(int pli, double time, Vector3D vout) {
        RVT rvt = new RVT(pli, time/Utils.DAY);
        rvt.setV(vout);
        return rvt;
    }

    static double bestY_ = 1E99;
                  
    @Override
    public double eval(double[] x) {
        double[] dvs = new double[9];
        double reso_penalty = 0;
        double t;
        double t0 = t = x[0]*Utils.DAY;
        double tof01 = x[1]*Utils.DAY;
        double tof23 = x[2]*Utils.DAY;
        double tof34 = x[3]*Utils.DAY;
        double[] beta = Arrays.copyOfRange(x, 4, 11);
        
        List<RVT> outs = new ArrayList<RVT>(); // trajectory orbits
        RVT start = new RVT(3, t/Utils.DAY);
        Vector3D[] vout_in = new Vector3D[2];
        double dvStart = mga(start, 3, 2, t, tof01, vout_in, outs); 
        if (dvStart > maxLaunchDV)
            dvs[0] = Math.max(0, dvStart - maxLaunchDV);
        
        int[][] resos = new int[6][];
        
        t += tof01;
        Resonance res1 = Resonance.resonance(2, t, vout_in[1], resos_[0], beta[0], safe_distance, outs);
        reso_penalty += res1._dt;
        resos[0] = res1._reso;
        
        t += res1.tof();
        RVT in = rvt(2, t, vout_in[1]);
        dvs[1] = mga(in, 2, 3, t, tof23, vout_in, outs); 
 
        t += tof23;
        in = rvt(3, t, vout_in[1]);
        dvs[2] = mga(in, 3, 2, t, tof34, vout_in, outs); 
        
        t += tof34;
        Resonance res = null;
        for (int r = 1; r < resos.length; r++) {
            Vector3D v_in = res != null ? res._vout : vout_in[1];
            res = Resonance.resonance(2, t, v_in, resos_[r], beta[r], safe_distance, outs);
            reso_penalty += res._dt;
            resos[r] = res._reso;
            t += res.tof();
        }
        RVT fin = fb_prop_rotate(2, t, res._vout, beta[6]);
        Kepler finKep = fin.kepler();
        
        // orbit should be as polar as possible, but we do not care about prograde/retrograde
        double corrected_inclination = Math.toDegrees(Math.abs(Math.abs(finKep.i() % Math.PI - Math.PI/2)));
        double final_perhelion = finKep.periapsis()/AU;
        double final_aphelion = finKep.apoapsis()/AU;
        
        double min_sun_distance = final_perhelion;
        double max_sun_distance = final_aphelion;
        
        for (int i = 0; i < outs.size()-1; i++) {
            RVT rvt0 = outs.get(i);
            RVT rvt1 = outs.get(i+1);
            double dt = rvt1.t() - rvt0.t();
            
            Kepler kep = rvt0.kepler();
            double period = kep.period(rvt0.mu());
            if (dt > period) {
                max_sun_distance = Math.max(max_sun_distance, kep.apoapsis()/AU);
                min_sun_distance = Math.min(min_sun_distance, kep.periapsis()/AU);
            }   
//            System.out.println("" + i + " " + Utils.r(dt/Utils.DAY) + " " +
//                    Utils.r(period/Utils.DAY) + " " + Utils.r(Math.toDegrees(kep.i())) +  " " + 
//                    Utils.r(min_sun_distance) + " " + Utils.r(max_sun_distance) + " " + rvt0); 
        }
        double distance_penalty = Math.max(0, min_dist_sun - min_sun_distance);  
        distance_penalty += Math.max(0, max_sun_distance - max_dist_sun);  

        if (final_perhelion < min_dist_sun)
            final_perhelion += 10*(min_dist_sun - final_perhelion);
        double time_val = (t - t0)/Utils.DAY;
        if (time_val > max_mission_time)
            time_val += 10*(time_val - max_mission_time);
        
        double value = 100*Utils.sum(dvs) + reso_penalty + 100*(corrected_inclination) + 
                5000*(final_perhelion-min_dist_sun) + 0.5*time_val + 50000 * distance_penalty;
        if (value < bestY_) {
             System.err.println("" + Utils.r(Math.toDegrees(finKep.i())) +  " " + 
                     Utils.r(final_perhelion) + " " + Utils.r(min_sun_distance) + " " + 
                     Utils.r(max_sun_distance) + " " + Utils.r(Utils.sum(dvs)) + " " + 
                     Utils.r(reso_penalty) + " " + Utils.r(time_val) + " " + 
                     Utils.r(resos) + " " + this);  
             bestY_ = value;
             _resos = resos;
             _outs = outs;
        }
        return value;
    }

    Solo optimize() {
        Utils.startTiming();
        Solo fit = create();
        fit.minimizeN(12800000, new Bite(6), 100000, 0, 31, 1E99);
//        CoordRetry.optimize(80000, this, new DE(), null, 1E99, 0, 2000, true);
//        CoordRetry.optimize(80000, this, new DECMA(), null, 1E99, 0, 2000, true);
//        CoordRetry.optimize(80000, this, new DeBite(), null, 1E99, 0, 2000, true);        
        System.out.println(fit);
        System.out.println(Utils.measuredMillis() + " ms");
        return fit;
    }
    
    void check_good_solution() {
        double[] x = new double[] {7454.820505282011, 399.5883816298621, 161.3293044402143, 336.35353340379817, 0.16706526043179085, -2.926263900573538, 
                2.1707384653871475, 3.068749728236526, 2.6458336313296913, 3.0472278514692377, 2.426804445518446};
        Solo solo = new Solo();
        solo.eval(x);        
    }

    public static void main(String[] args) throws FileNotFoundException {
        Log.setLog();
        Solo solo = new Solo();
//        solo.check_good_solution();
        solo.optimize();
    }

}

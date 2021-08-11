package fcmaes.core;

import java.io.IOException;

import org.hipparchus.geometry.euclidean.threed.Vector3D;

import com.nativeutils.NativeUtils;

public class Jni {

    static {
        try {
            NativeUtils.loadLibraryFromJar(("/natives/" + System.mapLibraryName("fcmaeslib")));
        } catch (IOException e1) {
            throw new RuntimeException(e1);
        }
        libraryLoaded = true;
    }
    
    public static boolean libraryLoaded = false;

    public static native int optimizeACMA(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
            int maxEvals, double stopValue, int popsize, int mu, double accuracy, long seed, int runid,
            boolean normalize, int update_gap, int workers);

    static native long initCmaes(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess, int popsize, int mu,
            double accuracy, long seed, int runid, boolean normalize, int update_gap);

    static native void destroyCmaes(long ptr);

    static native double[] askCmaes(long ptr);

    static native int tellCmaes(long ptr, double[] x, double y);

    static native double[] populationCmaes(long ptr);

    public static native int optimizeDE(Fitness fit, double[] lower, double[] upper, double[] result, int maxEvals,
            double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid, int workers);

    static native long initDE(Fitness fit, double[] lower, double[] upper, int popsize, double keep, double F,
            double CR, long seed, int runid);

    static native void destroyDE(long ptr);

    static native double[] askDE(long ptr);

    static native int tellDE(long ptr, double[] x, double y, int p);
    
    static native double[] populationDE(long ptr);
        
    public static native double[] optimizeMODE(Fitness fit, 
    		int dim, int nobj, int ncon,
    		double[] lower, double[] upper, 
    		int maxEvals, double stopfitness, int popsize, 
    		double keep, double F, double CR, 
    		double pro_c, double dis_c, double pro_m, double dis_m,
    	    boolean nsga_update, double pareto_update, int log_period,
    		long seed, int workers, int runid);
    
    static native long initMODE(Fitness fit, int dim, int nobj, int ncon,
    		double[] lower, double[] upper, 
    		int maxEvals, double stopfitness, int popsize, 
    		double keep, double F, double CR, 
    		double pro_c, double dis_c, double pro_m, double dis_m,
    	    boolean nsga_update, double pareto_update, int log_period,
    		long seed, int runid);

    static native void destroyMODE(long ptr);

    static native double[] askMODE(long ptr);

    static native int tellMODE(long ptr, double[] x, double[] y, int p);

    static native double[] populationMODE(long ptr);

    public static native int integrateF8(double[] y, double w, double dt, double step);
	
	public static native int integratePVthrust(double[] tpv, double veff, 
		    double ux, double uy, double uz, double dt, double step);

    public static native int integratePVgtocX(double[] tpv, double dt, double step);

    public static native int integratePVCtoc11(double[] tpv, double dt, double step, boolean dopri);

    public static native int integratePVCtoc11R(double[] tpv, double dt, double step, boolean dopri);

    public static native int optimizeLDE(Fitness fit, double[] lower, double[] upper, double[] guess, double[] sigma,
            int maxEvals, double stopfitness, int popsize, double keep, double F, double CR, long seed, int runid);

    public static native int optimizeGCLDE(Fitness fit, double[] lower, double[] upper, double[] guess, int maxEvals,
            double stopfitness, int popsize, double pbest, double F0, double CR0, long seed, int runid);

    public static native int optimizeLCLDE(Fitness fit, double[] lower, double[] upper, double[] guess, double[] sigma,
            int maxEvals, double stopfitness, int popsize, double pbest, double F0, double CR0, long seed, int runid);

    public static native int optimizeDA(Fitness fit, double[] lower, double[] upper, double[] guess, int maxEvals,
            int use_local_search, long seed, int runid);

    public static native int optimizeCLDE(Fitness fit, double[] lower, double[] upper, double[] guess, 
            int maxEvals, double stopfitness, int popsize, double pbest, double K1, double K2, long seed, int runid);

    public static native int optimizeBite(Fitness fit, double[] lower, double[] upper, double[] guess, int maxEvals,
            double stopfitness, int M, int stallLimit, long seed, int runid);

    public static native int optimizeCsma(Fitness fit, double[] lower, double[] upper, double[] sigma, double[] guess,
            int maxEvals, double stopfitness, int popsize, int stallLimit, long seed, int runid);
    
    public static native void planetEplC(int pli, double mjd2000, double[] r, double[] v);

//  data[0] = pl.compute_period(mjd2000);
//  data[1] = pl.get_mu_central_body();
//  data[2] = pl.get_mu_self();
//  data[3] = pl.get_radius();
//  data[4] = pl.get_safe_radius();

    public static native void planetDataC(int pli, double mjd2000, double[] data);

    public static native void iC2parC(double[] r0, double[] v0, double mu, double[] E);

    public static native void par2IC(double[] E, double mu, double[] r0, double[] v0);

    public static native void fb_prop(double[] v_in, double[] v_pla, double rp, double beta, double mu,
            /* out */double[] v_out);

    public static native double fb_vel(double[] v_rel_in, double[] v_rel_out, int planet); // returns dV
    
    public static native void propagate_lagrangian(double[] r0, double[] v0, double t, double mu);

    public static native double propagateTAYC(double[] r0_in, double[] v0_in, double dt, double[] c0_in, double m0,
            double mu, double veff, double log10tolerance, double log10rtolerance, double[] r, double[] v);

    public static native double propagateTAYJ2C(double[] r0_in, double[] v0_in, double dt, double[] c0_in, double m0,
            double mu, double veff, double j2rg2, double log10tolerance, double log10rtolerance, double[] r,
            double[] v);

    public static native double[] lambertProblem(double r1x, double r1y, double r1z, double r2x, double r2y, double r2z,
            double tf, boolean retro, int N, double mu);

    public static double[] lambertProblem(Vector3D r1, Vector3D r2, double tf, double mu, boolean retro,
            int N) {
        return lambertProblem(r1.getX(), r1.getY(), r1.getZ(), r2.getX(), r2.getY(), r2.getZ(), tf, retro, N, mu);
    }

    public static Vector3D[] bestLambert(double[] lamb, Vector3D vFrom) {
        Vector3D vOutMin = null;
        Vector3D vInMin = null;
        double minDV = 1E99;
        for (int i = 0; i < lamb.length; i += 6) {
            Vector3D vOut = new Vector3D(lamb[i + 0], lamb[i + 1], lamb[i + 2]);
            double dv = vOut.subtract(vFrom).getNorm();
            if (dv < minDV) {
                minDV = dv;
                vOutMin = vOut;
                vInMin = new Vector3D(lamb[i + 3], lamb[i + 4], lamb[i + 5]);
            }
        }
        if (vInMin == null)
            return null;
        return new Vector3D[] { vOutMin, vInMin };
    }

    public static Vector3D[] bestLambert(double[] lamb, int pli, double mjd2000, Vector3D vFrom) {
        Vector3D vOutMin = null;
        Vector3D vInMin = null;
        double minDV = 1E99;
        double[] r = new double[3];
        double[] v = new double[3];
        Jni.planetEplC(pli, mjd2000, r, v);
        Vector3D vpl = Utils.vector(v);
        double[] v_rel_in = Utils.array(vFrom.subtract(vpl));
        for (int i = 0; i < lamb.length; i += 6) {
            Vector3D vOut = new Vector3D(lamb[i + 0], lamb[i + 1], lamb[i + 2]);
            double[] v_rel_out = Utils.array(vOut.subtract(vpl));
            double dv = Jni.fb_vel(v_rel_in, v_rel_out, pli);
            if (dv < minDV) {
                minDV = dv;
                vOutMin = vOut;
                vInMin = new Vector3D(lamb[i + 3], lamb[i + 4], lamb[i + 5]);
            }
        }
        if (vInMin == null)
            return null;
        return new Vector3D[] { vOutMin, vInMin };
    }

}

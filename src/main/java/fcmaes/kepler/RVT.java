package fcmaes.kepler;

import org.hipparchus.geometry.euclidean.threed.Vector3D;

import fcmaes.core.Jni;
import fcmaes.core.Utils;

/**
 * Works not on Windows! Use the Linux subsystem for Windows there.
 * 
 * Orbit represented by r (position), v (velocity), t (time) and mu.
 */

public class RVT  {

    public double[] rvt;

    public RVT() {
        rvt = new double[9];
    }

    public RVT(RVT other) {
        rvt = other.rvt.clone();
    }

    public RVT(double mu, double t) {
        rvt = new double[9];
        setMu(mu);
        setT(t);
    }

    public RVT(Vector3D r, Vector3D v, double mu, double t) {
        rvt = new double[9];
        setR(r);
        setV(v);
        setMu(mu);
        setT(t);
    }

    public RVT(Vector3D r, Vector3D v, double mu, double t, double m) {
        rvt = new double[9];
        setR(r);
        setV(v);
        setMu(mu);
        setT(t);
        setM(m);
    }
 
    public RVT(int pli, double mjd2000) {
        rvt = new double[9];
        double[] r = new double[3];
        double[] v = new double[3];
        Jni.planetEplC(pli, mjd2000, r, v);
        double[] data = new double[5];
        Jni.planetDataC(pli, mjd2000, data);
        setR(r);
        setV(v);
        setMu(data[1]); // mu central body
        setT(mjd2000*Utils.DAY);
    }
    
    public RVT(Kepler kepler, double mu) {
        rvt = new double[9];
        double[] r = new double[3];
        double[] v = new double[3];
        Jni.par2IC(kepler.kep, mu, r, v);
        setR(r);
        setV(v);
        setMu(mu);
    }
    
    public Kepler kepler() {
        return new Kepler(this);
    }
    
    public double period() {
        return kepler().period(mu());
    }

    double[] deltaV(RVT other) { // returns dv, dv1, dv2
        Vector3D dPos = other.r().subtract(r());
        double duration = other.t() - t();
        Vector3D vOut = dPos.scalarMultiply(1.0 / duration);
        double dv1 = vOut.subtract(v()).getNorm();
        double dv2 = other.v().subtract(vOut).getNorm();
       return new double[] {dv1 + dv2, dv1, dv2};
    }
    
    public void applyDV(Vector3D dv) {
        if (dv.getNorm() > 0)
            setV(v().add(dv));
    }

    public void propagate_kepler(double dt) {
        Kepler orb = new Kepler(this);
        orb.propagate_kepler(dt, mu());
        double t = t() + dt;
        double m = m();
        rvt = new RVT(orb, mu()).rvt;
        setT(t);
        setM(m);
    }

    public void propagate_lagrangian(double dt) {
        try {
            double[] r0 = rarr();
            double[] v0 = varr();
            Jni.propagate_lagrangian(r0, v0, dt, mu());
            setR(r0);
            setV(v0);
            setT(t() + dt);
        } catch (Exception ex) {
            System.err.println(ex);
        }
    }

    public void propagate_taylor(double dt, Vector3D thrust, double veff) {
        try {
            double[] r = rarr();
            double[] v = varr();
            double mass = Jni.propagateTAYC(rarr(), varr(), dt, Utils.array(thrust), m(), mu(), veff, -10, -8, r, v);
            for (int i = 0; i < 3; i++) {
                if (Double.isInfinite(r[i]) || Double.isInfinite(v[i]) || Double.isNaN(r[i]) || Double.isNaN(v[i])) {
                    System.err.println(r[i] + " " + v[i]);
                }
            }
            setR(r);
            setV(v);
            setT(t() + dt);
            setM(mass);
        } catch (Exception ex) {
            System.err.println(ex);
        }
    }

    public void propagate_taylor_J2(double dt, Vector3D thrust, double veff, double j2rg2) {
        try {
            double[] r = rarr();
            double[] v = varr();
            double mass = Jni.propagateTAYJ2C(rarr(), varr(), dt, Utils.array(thrust), m(), mu(), veff, j2rg2, -10, -8,
                    r, v);
            for (int i = 0; i < 3; i++) {
                if (Double.isInfinite(r[i]) || Double.isInfinite(v[i]) || Double.isNaN(r[i]) || Double.isNaN(v[i])) {
                    System.err.println(r[i] + " " + v[i]);
                }
            }
            setR(r);
            setV(v);
            setT(t() + dt);
            setM(mass);
        } catch (Exception ex) {
            System.err.println(ex);
        }
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("r = " + Utils.r(r()) + " ");
        sb.append("v = " + Utils.r(v()) + " ");
        sb.append("t = " + Utils.r(t() / Utils.DAY) + " ");
        sb.append("m = " + Utils.r(m()));
        return sb.toString();
    }

    public Vector3D r() {
        return new Vector3D(rvt[0], rvt[1], rvt[2]);
    }

    public Vector3D v() {
        return new Vector3D(rvt[3], rvt[4], rvt[5]);
    }

    public double[] rarr() {
        return new double[] { rvt[0], rvt[1], rvt[2] };
    }

    public double[] varr() {
        return new double[] { rvt[3], rvt[4], rvt[5] };
    }

    public double mu() {
        return rvt[6];
    }

    public double t() {
        return rvt[7];
    }

    public double m() {
        return rvt[8];
    }

    public void setR(Vector3D r) {
        rvt[0] = r.getX();
        rvt[1] = r.getY();
        rvt[2] = r.getZ();
    }

    public void setR(double[] r) {
        rvt[0] = r[0];
        rvt[1] = r[1];
        rvt[2] = r[2];
    }

    public void setV(Vector3D v) {
        rvt[3] = v.getX();
        rvt[4] = v.getY();
        rvt[5] = v.getZ();
    }

    public void setV(double[] v) {
        rvt[3] = v[0];
        rvt[4] = v[1];
        rvt[5] = v[2];
    }

    public void setMu(double mu) {
        rvt[6] = mu;
    }

    public void setT(double t) {
        rvt[7] = t;
    }

    public void setM(double m) {
        rvt[8] = m;
    }

    double R() {
        return r().getNorm();
    }
    
    public void applyDV(Vector3D dv, double G0, double isp) {
        if (dv.getNorm() > 0) {
            setV(v().add(dv));
            double m = m() * Math.exp(-dv.getNorm() / (G0 * isp));
            setM(m);
        }
    }
    
    public boolean valid() {
        return !Utils.isNaN(rvt);
    }

    double[] computeLambertProblem(RVT to, boolean retro, int revs) {
        Vector3D Ri = r();
        Vector3D Rf = to.r();
        double dt = to.t() - t();
        return Jni.lambertProblem(Ri, Rf, dt, mu(), retro, revs);
    }
   
    // lowest dv lambert transfer
    public Vector3D[] bestLambert(RVT to, boolean retro, int revs) {
        double[] lamb = computeLambertProblem(to, retro, revs);
        return Jni.bestLambert(lamb, v());
    }
    
    // lowest dv lambert transfer with GA at planet pli
    public Vector3D[] bestLambert(RVT to, int pli, boolean retro, int revs) {
        double[] lamb = computeLambertProblem(to, retro, revs);
        return Jni.bestLambert(lamb, pli, t()/Utils.DAY, v());
    }

}

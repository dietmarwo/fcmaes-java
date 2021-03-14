package fcmaes.kepler;

import org.hipparchus.util.FastMath;

import fcmaes.core.Jni;
import fcmaes.core.Utils;

/**
 * Orbit represented by keplarian parameters a, e, i, OM, om, ea.
 */

public class Kepler {

    public double[] kep;

    public Kepler() {
        kep = new double[6];
    }

    public Kepler(Kepler other) {
        kep = other.kep.clone();
    }

    public Kepler(RVT tpv) {
        kep = new double[6];
        Jni.iC2parC(tpv.rarr(), tpv.varr(), tpv.mu(), kep);
    }

    public Kepler(double[] kep) {
        this.kep = kep.clone();
    }

     // Orbit relative to M_at_epoch
    public Kepler(double a, double e, double i, double OM, double om, double M_at_epoch, double mu, double dt) {
        this.kep = new double[] { a, e, i, OM, om, 0 };
        setM0(M_at_epoch, dt, mu);
    }

    public Kepler(double a, double e, double i, double OM, double om, double ea, double mu) {
        this.kep = new double[] { a, e, i, OM, om, ea };
    }

    public boolean valid() {
        return !Utils.isNaN(kep);
    }

    public RVT tpv(double mu) {
        return new RVT(this, mu);
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("a = " + Utils.r(a()) + " ");
        sb.append("e = " + Utils.r(e()) + " ");
        sb.append("i = " + Utils.r(i()) + " ");
        sb.append("OM = " + Utils.r(OM()) + " ");
        sb.append("om = " + Utils.r(om()) + " ");
        sb.append("ea = " + Utils.r(ea()) + " ");
        return sb.toString();
    }
    
    public void propagate_kepler(double dt, double mu) {
        double M = 0;
        if (e() < 1) {
            M = ea() - e() * FastMath.sin(ea());
            M += FastMath.sqrt(mu / (a() * a() * a())) * dt;
        } else {
            M = e() * FastMath.tan(ea()) - FastMath.log(FastMath.tan(0.5 * ea() + 0.25 * FastMath.PI));
            M += FastMath.sqrt(FastMath.abs(mu / (a() * a() * a()))) * dt;
        }
        setEa(mean2eccentric(M));
    }

    // http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.xml
    public double mean2ta(final double M) {
        double Mnorm = M % (2 * FastMath.PI);
        double E = keplerStart3(e(), Mnorm);
        int count = 0;
        double delta;
        do {
            delta = eps3(e(), Mnorm, E);
            E -= delta;
        } while (FastMath.abs(delta) > 1.0e-12 && count++ < 50);
        return 2.0 * FastMath.atan(FastMath.tan(0.5 * E) * FastMath.sqrt((1.0 + e()) / (1.0 - e())));
    }

    public double eccentric2ta(double E) {
        return 2.0 * FastMath.atan(FastMath.tan(0.5 * E) * FastMath.sqrt((1.0 + e()) / (1.0 - e())));
    }

    public double getTrueAnomaly(double M0, double dt, double mu) {
        double M = M0 + dt * FastMath.sqrt(mu / (a() * a() * a()));
        return mean2ta(M);
    }

    public double getEccentricAnomaly(double M0, double dt, double mu) {
        double M = M0 + dt * FastMath.sqrt(mu / (a() * a() * a()));
        return mean2eccentric(M);
    }

    public double getEccentricAnomaly2(double M0, double dt, double mu) {
        double ta = getTrueAnomaly(M0, dt, mu);
        final double beta = e() / (1 + FastMath.sqrt((1 - e()) * (1 + e())));
        return ta - 2 * FastMath.atan(beta * FastMath.sin(ta) / (1 + beta * FastMath.cos(ta)));
    }

    double mean2eccentric(final double M) {
        double Mnorm = M % (2 * FastMath.PI);
        double E = keplerStart3(e(), Mnorm);
        int count = 0;
        double delta;
        do {
            delta = eps3(e(), Mnorm, E);
            E -= delta;
        } while (FastMath.abs(delta) > 1.0e-12 && count++ < 50);
        return ((E + FastMath.PI) % (2 * FastMath.PI)) - FastMath.PI;
    }

    static double keplerStart3(double e, double M) {
        double t34 = e * e;
        double t35 = e * t34;
        double t33 = FastMath.cos(M);
        return M + (-0.5 * t35 + e + (t34 + 1.5 * t33 * t35) * t33) * FastMath.sin(M);
    }

    static double eps3(double e, double M, double x) {
        double t1 = FastMath.cos(x);
        double t2 = -1.0 + e * t1;
        double t3 = FastMath.sin(x);
        double t4 = e * t3;
        double t5 = -x + t4 + M;
        double t6 = t5 / ((0.5 * t5) * t4 / t2 + t2);
        return t5 / ((0.5 * t3 - t1 * t6 / 6.0) * e * t6 + t2);
    }

    public double minRadius() {
        return a() * (1.0 - e());
    }

    public double maxRadius() {
        return a() * (1.0 + e());
    }

    public double meanMotion(double mu) {
        return FastMath.sqrt(mu / (a() * a() * a()));
    }

    public double period(double mu) {
        return 2.0 * FastMath.PI / meanMotion(mu);
    }

    public double periapsis() {
        return (1.0 - e()) * a();
    }

    public double apoapsis() {
        return (1.0 + e()) * a();
    }

    public static double periapsis(RVT tpv) {
        Kepler orb = new Kepler(tpv);
        return orb.periapsis();
    }

    public double a() {
        return kep[0];
    }

    public double e() {
        return kep[1];
    }

    public double i() {
        return kep[2];
    }

    public double OM() {
        return kep[3];
    }

    public double om() {
        return kep[4];
    }

    public double ea() {
        return kep[5];
    }

    public void setA(double a) {
        kep[0] = a;
    }

    public void setE(double e) {
        kep[1] = e;
    }

    public void setI(double i) {
        kep[2] = i;
    }

    public void setOM(double OM) {
        kep[3] = OM;
    }

    public void setom(double om) {
        kep[4] = om;
    }

    public void setEa(double ea) {
        kep[5] = ea;
    }

    public void setM0(double M0, double dt, double mu) {
        double ea = getEccentricAnomaly(M0, dt, mu);
        setEa(ea);
    }

}

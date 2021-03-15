package fcmaes.kepler;

import java.util.List;

import org.hipparchus.geometry.euclidean.threed.Vector3D;

import fcmaes.core.Jni;
import fcmaes.core.Utils;

/**
 * Works not on Windows! Use the Linux subsystem for Windows there.
 *  
 * Resonance Orbit at planet, selects the resonance alternative with lowest timing error _dt.
 */

public class Resonance {

    // input
    RVT _planet;
    double[] _data;
    Vector3D _vin;
    double[] _v_in;
    public int[][] _resos;
    
    // output
    public Vector3D _vout;
    public double[] _v_pla;
    public double _beta;
    public double _rp;
    public double _period;
    public int _index;
    public double _dt;
     
    public Resonance(int pli, double time, Vector3D vin, int[][] resos, double safe_distance) {
        double mjd2000 = time/Utils.DAY;
        _planet = new RVT(pli, mjd2000); 
        _data = new double[5];
        Jni.planetDataC(pli, mjd2000, _data);
        _vin = vin;
        _v_in = Utils.array(vin);
        _v_pla = _planet.varr();
        _rp = _data[3] + safe_distance; // safe radius 
        _period = _data[0];
        _resos = resos;
    }
    public double select(double beta, List<RVT> outs) {
        double[] v_out = new double[3];
        Jni.fb_prop(_v_in, _v_pla, _rp, beta, _data[2], /*out*/v_out);
        RVT out = new RVT(_planet);
        out.setV(Utils.vector(v_out));
        outs.add(out);
        double period = out.period();
        double dt = Double.MAX_VALUE;
        int resoI = 0;
        for (int i = 0; i < _resos.length; i++) {
            double target_per = _period * _resos[i][1] / _resos[i][0];
            double diff = Math.abs(period - target_per);
            if (diff < dt) {
                resoI = i;
                dt = diff;
            }           
        }
        _vout = Utils.vector(v_out);
        _beta = beta;
        _index = resoI;
        _dt = dt;
        return dt;
    }

    public int[] selected() {
        return _resos[_index];
    }

    public double tof() {
        return _period * selected()[1];
    }
        
    public static Resonance resonance(int pli, double time, Vector3D vin, int[][] resos, 
            double beta, double safe_distance, List<RVT> outs, List<Double> dvs) {
        Resonance res = new Resonance(pli, time, vin, resos, safe_distance);
        res.select(beta, outs);
        dvs.add(0.0);
        return res;
    }
}

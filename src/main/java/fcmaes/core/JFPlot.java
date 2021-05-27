/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.Plot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class JFPlot extends JFrame {

    private static final long serialVersionUID = 2443898856454800479L;

    private JFreeChart chart;
    private ChartPanel panel;
    int sizeX;
    int sizeY;

    public JFPlot(XYDataset dataset, int sizeX, int sizeY) {
        this.sizeX = sizeX;
        this.sizeY = sizeY;

        chart = ChartFactory.createScatterPlot("fitness", "fit1", "fit2", dataset,
                org.jfree.chart.plot.PlotOrientation.VERTICAL, true, true, false);
        panel = new ChartPanel(chart);
        panel.setMouseZoomable(false);
        this.setSize(sizeX, sizeY);
        setContentPane(panel);
    }

    public JFPlot(double[][] data, int sizeX, int sizeY) {
        this(dataSet(data), sizeX, sizeY);
    }

    static private XYDataset dataSet(double[][] data) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        XYSeries series1 = new XYSeries("fitness");
        for (int i = 0; i < data.length; i++)
            series1.add(data[i][0], data[i][1]);
        dataset.addSeries(series1);
        return dataset;
    }

    void setBackgroundImage(String fname) {
        ImageIcon icon = new ImageIcon(fname);
        chart.setBackgroundImage(icon.getImage());
        Plot plot = chart.getPlot();
        plot.setBackgroundAlpha(0.0f);
        // myPlot.setForegroundAlpha(0.9f);
    }

    public void writeAsImage(String name) {
        setVisible(true);
        SwingUtilities.invokeLater(() -> {
            try {
                BufferedImage image = new BufferedImage(sizeX, sizeY, BufferedImage.TYPE_INT_ARGB);
                Graphics2D g2 = image.createGraphics();
                g2.setRenderingHint(JFreeChart.KEY_SUPPRESS_SHADOW_GENERATION, true);
                Rectangle r = new Rectangle(0, 0, sizeX, sizeY);
                chart.draw(g2, r);
                File f = new File(name + ".png");
                BufferedImage chartImage = chart.createBufferedImage(sizeX, sizeY, null);
                ImageIO.write(chartImage, "png", f);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        });
        setVisible(false);
        dispose();
    }

}

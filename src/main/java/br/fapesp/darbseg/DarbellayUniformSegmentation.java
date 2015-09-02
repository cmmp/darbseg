/*
 * This code separates the non-uniform regions of a space
 * using the Darbellay segmentation approach.
 */
package br.fapesp.darbseg;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Paint;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartColor;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.renderer.xy.XYStepRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import br.fapesp.myutils.MyUtils;

/**
 * Code based on the paper:
 * Hudson, J. (2006). Signal processing using mutual information. 
 * Signal Processing Magazine, IEEE, (November), 50–54. 
 *
 * @author Cássio M. M. Pereira <cassiomartini@gmail.com>
 * Created on 29/08/2015
 * 
 */
public class DarbellayUniformSegmentation {
	
	private static final boolean DEBUG = true;
	private static final boolean GUIDEBUG = true;
	
	private static JFreeChart chart;
	private static ChartFrame frame;
	
	private static int keyCounter = 0;
	
	/**
	 * Darbellay segmentation
	 * 
	 * @param data 2-d matrix with examples on rows
	 * @param nbreaks number of quantile breaks to make
	 * @param pcrit probability for critical value, e.g., 0.95; 0.99
	 * @return labels for uniform regions identified
	 */
    public static int[] darbellay(double[][] data, int nbreaks, double pcrit) {
    	
    	ChiSquaredDistribution cd = new ChiSquaredDistribution((nbreaks + 1) * (nbreaks + 1) - 1);
    	double crit = cd.inverseCumulativeProbability(pcrit);
    	
    	if(GUIDEBUG)
    		plotMain(data);
    	
    	List<Integer> l = new ArrayList<Integer>();
    	for (int i = 0; i < data.length; i++)
    		l.add(i);
    	List<List<Integer>> partition = darbellay(l, new Array2DRowRealMatrix(data), nbreaks, pcrit, crit);
    	
    	if(DEBUG)
    		System.out.println("Final partition: " + partition);
    		
    	// create labels from partition:
    	int K = 1;
    	int[] lbls = new int[data.length];
    	for(int i = 0; i < partition.size(); i++) {
    		List<Integer> pts = partition.get(i);
    		for(Integer j : pts)
    			lbls[j] = K;
    		K++;
    	}
    	
    	if (DEBUG) {
    		System.out.println("final labels:");
    		System.out.println(Arrays.toString(lbls));
    	}
    	
    	return lbls;
    }
	
	private static List<List<Integer>> darbellay(List<Integer> pts, RealMatrix origmat, int nbreaks, double pcrit, double crit) {
		
		List<List<Integer>> ret = new ArrayList<List<Integer>>();
		
		int m = origmat.getColumnDimension();
		
		if (m != 2)
			throw new RuntimeException("impl only for matrices in 2d space!");
		
		double[][] d = new double[pts.size()][2];
		for (int i = 0; i < pts.size(); i++)
			d[i] = origmat.getRow(pts.get(i));
		
		RealMatrix data = new Array2DRowRealMatrix(d);
		int N = data.getRowDimension();
		
		if (DEBUG) {
			System.out.println("--------------------------------------------------------------------------------");
			System.out.println("darb called with matrix (" + N + "," + m + "):");
			System.out.println(data);
		}	
		
		// find the quantiles on each dimension:
		double[][] quant = new double[nbreaks][2];
		DescriptiveStatistics ds0 = new DescriptiveStatistics(data.getColumn(0));
		DescriptiveStatistics ds1 = new DescriptiveStatistics(data.getColumn(1));
		
		double[] pctls = MyUtils.computePercentilesFromNbreaks(nbreaks);
		
		for (int i = 0; i < nbreaks; i++) {
			quant[i][0] = ds0.getPercentile(pctls[i]);
			quant[i][1] = ds1.getPercentile(pctls[i]);
		}
		
		int nchild = (nbreaks + 1) * (nbreaks + 1);
		
		@SuppressWarnings("unchecked")
		List<Integer>[] child = new ArrayList[nchild];
		
		for (int i = 0; i < nchild; i++)
			child[i] = new ArrayList<Integer>();
		
		double maxX = ds0.getMax();
		double maxY = ds1.getMax();
		
		for (int i = 0; i < N; i++) {
			double x, y;
			x = data.getEntry(i, 0); y = data.getEntry(i, 1);
			
			// test on which bin of the double "histogram" (x,y) falls
			for (int j = 0; j <= nbreaks; j++) {
				double xcmp = j < nbreaks ? quant[j][0] : maxX;
				
				if (x <= xcmp) {
					for (int k = 0; k <= nbreaks; k++) {
						double ycmp = k < nbreaks ? quant[k][1] : maxY;
						if (y <= ycmp) {
							child[j + k * nbreaks + k].add(pts.get(i));
							break;
						}
					}
					break;
				}
			}				
		}
		
		if (DEBUG) {
			for (int i = 0; i < nchild; i++)
				System.out.println("n(ch" + i + ") = " + child[i].size());
		}
		
		if (GUIDEBUG) {
			plotRegion(data.getData(), quant);
			Scanner scan = new Scanner(System.in);
			System.out.println("Press enter to continue...");
			String input = scan.nextLine();
		}
		
		double mean = N / nchild;
		
		// compute the test statistic:
		double T = 0;
		double diff;
		
		for (int i = 0; i < nchild; i++) {
			diff = mean - child[i].size();
			T += diff * diff;
		}

		if (DEBUG) {
			System.out.println("calculated T = " + T + "; X^2(" + pcrit + ","+ ((nbreaks + 1) * (nbreaks + 1) - 1) + "df) crit = " + crit);
		}
			
		if (T < crit) {
			// the sample already comes from a uniform distribution
			if (DEBUG)
				System.out.println(">>> adicionando lista: " + pts);
			
			ret.add(pts);
		} else {
			// keep dividing
			for (int i = 0; i < nchild; i++)
				if(child[i].size() > 1)
					ret.addAll(darbellay(child[i], origmat, nbreaks, pcrit, crit));
		}
		
		return ret;
	}
	
    
    
    public static void main(String args[]) throws IOException {
    	
		File csvData = new File("C:/Users/Cássio/workspace/darbseg/resources/Y2.csv"); // topology data set
//    	File csvData = new File("C:/Users/Cássio/workspace/darbseg/resources/sample.csv"); // example given in the paper
//    	File csvData = new File("C:/Users/Cássio/workspace/darbseg/resources/sample3.csv"); // gaussians
    	
        CSVParser parser = CSVParser.parse(csvData, Charset.defaultCharset(), CSVFormat.RFC4180.withDelimiter(' ').withSkipHeaderRecord().withIgnoreEmptyLines());
        
        List<CSVRecord> list = parser.getRecords();
        
        double[][] data = new double[list.size()][list.get(0).size()];
        
        for (int i = 0; i < list.size(); i++) {
        	CSVRecord r = list.get(i);
        	for (int j = 0; j < r.size(); j++)
        		data[i][j] = Double.parseDouble(r.get(j));
        }
        
        darbellay(data, 3, 0.99);
        
    }
    
    private static void plotMain(double[][] pts) {
    	List<Paint> colors = Arrays.asList(ChartColor.createDefaultPaintArray());
		Collections.reverse(colors);
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		
		XYSeries p = new XYSeries("Points");
		for (int i = 0; i < pts.length; i++)
			p.add(pts[i][0], pts[i][1]);
		dataset.addSeries(p);
		
		// create chart:
		chart = ChartFactory.createXYLineChart("Plot", "x", "y", dataset);
		XYLineAndShapeRenderer xr = (XYLineAndShapeRenderer) chart.getXYPlot().getRenderer();
		xr.setSeriesLinesVisible(0, false);
		xr.setSeriesShapesVisible(0, true);
		chart.removeLegend();
		
		frame = new ChartFrame("Plot window", chart);
		frame.pack();
		frame.setDefaultCloseOperation(ChartFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);		
    }
    
    /**
	 * Plot the data set and set of medoids
	 * @param pts points matrix
	 * @param quant quantiles matrix
	 */
	private static void plotRegion(double[][] pts, double[][] quant) {
		RealMatrix mat = new Array2DRowRealMatrix(pts);
		DescriptiveStatistics ds0 = new DescriptiveStatistics(mat.getColumn(0));
		DescriptiveStatistics ds1 = new DescriptiveStatistics(mat.getColumn(1));
		
		double xmin = ds0.getMin(); double xmax = ds0.getMax();
		double ymin = ds1.getMin(); double ymax = ds1.getMax();		
		
		for (int i = 0; i < quant.length; i++) {
			// plot x quant:
			XYSeries xy1 = new XYSeries(++keyCounter);
			xy1.add(quant[i][0], ymin);
			xy1.add(quant[i][0], ymax);			
			
			XYSeriesCollection set = (XYSeriesCollection) chart.getXYPlot().getDataset();
			set.addSeries(xy1);
			chart.getXYPlot().getRenderer().setSeriesPaint(keyCounter, Color.black);
			
			XYSeries xy2 = new XYSeries(++keyCounter);
			xy2.add(xmin, quant[i][1]);
			xy2.add(xmax, quant[i][1]);			
			
			set.addSeries(xy2);
			chart.getXYPlot().getRenderer().setSeriesPaint(keyCounter, Color.black);
			
		}
		
		frame.pack();
		frame.repaint();
	}
    
    
}

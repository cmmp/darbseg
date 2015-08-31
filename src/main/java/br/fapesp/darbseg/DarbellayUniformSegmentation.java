/*
 * This code separates the non-uniform regions of a space
 * using the Darbellay segmentation approach.
 */
package br.fapesp.darbseg;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * Code based on the paper:
 * Hudson, J. (2006). Signal processing using mutual information. 
 * Signal Processing Magazine, IEEE, (November), 50–54. 
 *
 * @author Cássio M. M. Pereira <cassiomartini@gmail.com>
 * 29/08/2015
 * 
 */
public class DarbellayUniformSegmentation {
	
	private static final boolean DEBUG = true;
	
	private static final double PCRIT = 0.99;
	private static final ChiSquaredDistribution CD = new ChiSquaredDistribution(3);
    private static final double CRIT = CD.inverseCumulativeProbability(PCRIT);
    
    public static int[] darbellay(double[][] data) {
    	List<Integer> l = new ArrayList<Integer>();
    	for (int i = 0; i < data.length; i++)
    		l.add(i);
    	List<List<Integer>> partition = darbellay(l, new Array2DRowRealMatrix(data));
    	
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
	
	public static List<List<Integer>> darbellay(List<Integer> pts, RealMatrix origmat) {
		
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
		
		// find the median on each dimension:
		double md0, md1;
		DescriptiveStatistics ds0 = new DescriptiveStatistics(data.getColumn(0));
		DescriptiveStatistics ds1 = new DescriptiveStatistics(data.getColumn(1));
		
		md0 = ds0.getPercentile(50);
		md1 = ds1.getPercentile(50);
		
		List<Integer>[] child = new ArrayList[4];
		
		for (int i = 0; i < 4; i++)
			child[i] = new ArrayList<Integer>();
		
		for (int i = 0; i < N; i++) {
			double x, y;
			x = data.getEntry(i, 0); y = data.getEntry(i, 1);
			
			if (x <= md0 && y <= md1)
				child[2].add(pts.get(i));
			else if(x <= md0 && y >= md1)
				child[1].add(pts.get(i));
			else if(x >= md0 && y <= md1)
				child[3].add(pts.get(i));
			else
				child[0].add(pts.get(i));			
		}
		
		if (DEBUG) {
			for (int i = 0; i < 4; i++)
				System.out.println("n(ch" + i + ") = " + child[i].size());
		}
		
		double mean = N / 4.0;
		
		// compute the test statistic:
		double T = 0;
		double diff;
		
		for (int i = 0; i < 4; i++) {
			diff = mean - child[i].size();
			T += diff * diff;
		}

		if (DEBUG) {
			System.out.println("calculated T = " + T + "; X^2(" + PCRIT + ",3df) crit = " + CRIT);
		}
			
		if (T < CRIT) {
			// the sample already comes from a uniform distribution
			if (DEBUG)
				System.out.println(">>> adicionando lista: " + pts);
			
			ret.add(pts);
		} else {
			// keep dividing
			for (int i = 0; i < 4; i++)
				if(child[i].size() > 1)
					ret.addAll(darbellay(child[i], origmat));
		}
		
		return ret;
	}
	
    
    
    public static void main(String args[]) throws IOException {
    	
//		File csvData = new File("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/Y2.csv");
    	File csvData = new File("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/sample.csv"); // example given in the paper
//    	File csvData = new File("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/sample2.csv"); 
//    	File csvData = new File("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/sample3.csv");
    	
        CSVParser parser = CSVParser.parse(csvData, Charset.defaultCharset(), CSVFormat.RFC4180.withDelimiter(' ').withSkipHeaderRecord().withIgnoreEmptyLines());
        
        List<CSVRecord> list = parser.getRecords();
        
        double[][] data = new double[list.size()][list.get(0).size()];
        
        for (int i = 0; i < list.size(); i++) {
        	CSVRecord r = list.get(i);
        	for (int j = 0; j < r.size(); j++)
        		data[i][j] = Double.parseDouble(r.get(j));
        }
        
//        for(int i = 0; i < list.size(); i++)
//        	System.out.println(Arrays.toString(data[i]));
        
//        List<Integer> l1 = new ArrayList<>();
//        l1.add(1); l1.add(2);
//        
//        List<Integer> l2 = new ArrayList<>();
//        l2.add(3); l2.add(4);
//        
//        List<List<Integer>> lists = new ArrayList<>();
//        lists.add(l1);
//        lists.add(l2);
//        
//        List<List<Integer>> lists2 = new ArrayList<>();
//        lists2.addAll(lists);
//        
//        System.out.println(lists2);
        
        darbellay(data);
        
    }
    
    
}

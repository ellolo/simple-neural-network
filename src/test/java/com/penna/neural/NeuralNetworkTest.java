package com.penna.neural;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.penna.neural.core.NeuralNetwork;
import com.penna.neural.functions.ActivationFunctions;
import com.penna.neural.functions.CostFunctions;

public class NeuralNetworkTest {
	
	@Test
	public void termSpikeDetectorRealTimeJobTest() throws Exception {
		DoubleMatrix weightsLayer1 = new DoubleMatrix(new double[][]{
				{4.0d, 3.0d, -2.5d},
				{1.5d, 0.0d, -1.0d}});
		DoubleMatrix weightsLayer2 = new DoubleMatrix(new double[][]{
				{1.0d, -2.5d},
				{2.0d,  3.5d},
				{0.0d,  1.0d},
				{-2.0d, 1.5d}});
		DoubleMatrix biasesLayer1 = new DoubleMatrix( new double[][]{{1.0}, {-0.5}});
		DoubleMatrix biasesLayer2 = new DoubleMatrix( new double[][]{{-3.0}, {1.5}, {0.0}, {-1.5}});
		DoubleMatrix[] weights = {weightsLayer1, weightsLayer2};
		DoubleMatrix[] biases = {biasesLayer1, biasesLayer2};
		NeuralNetwork nnTest = new NeuralNetwork(new int[]{3,2,4}, biases, weights, CostFunctions.QUADRATIC, ActivationFunctions.SIGMOID);
		DoubleMatrix output = nnTest.feedForward(new DoubleMatrix(new double[][]{{2.0d},{-1.5d},{1.0d}}));
		System.out.println(output.toString("%.3f"));
	}
}

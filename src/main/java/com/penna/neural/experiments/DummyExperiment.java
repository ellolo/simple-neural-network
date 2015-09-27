package com.penna.neural.experiments;

import org.jblas.DoubleMatrix;

import com.penna.neural.core.Dataset;
import com.penna.neural.core.Instance;
import com.penna.neural.core.NeuralNetwork;
import com.penna.neural.exceptions.NetworkInitializationException;
import com.penna.neural.functions.ActivationFunctions;
import com.penna.neural.functions.CostFunctions;
import com.penna.neural.utils.DoubleMatrixUtils;

/**
 * Dummy experiment create fictitiuos instances and trains a network. Goal of
 * this example is to expose an example code of how to use the neural network
 * class.
 * 
 * @author mpennacchiotti
 * 
 */
public class DummyExperiment {

    public static void main(String[] argv) throws NetworkInitializationException {
        // Creating fictitious training instance
        Instance trainingIns1 = new Instance(new DoubleMatrix(new double[][] { { 1.0 }, { -0.5 },
                { 3.0 } }), new DoubleMatrix(
                new double[][] { { 0.1 }, { 0.8 }, { 0.05 }, { 0.05 } }));
        Instance trainingIns2 = new Instance(new DoubleMatrix(new double[][] { { 0.0 }, { 7.0 },
                { -0.5 } }),
                new DoubleMatrix(new double[][] { { 0.0 }, { 0.0 }, { 0.1 }, { 0.9 } }));
        Instance trainingIns3 = new Instance(new DoubleMatrix(new double[][] { { 1.5 }, { -1.0 },
                { 5.0 } }), new DoubleMatrix(new double[][] { { 0.0 }, { 0.9 }, { 0.1 }, { 0.0 } }));
        Instance trainingIns4 = new Instance(new DoubleMatrix(new double[][] { { 0.0 }, { 6.0 },
                { 0.0 } }), new DoubleMatrix(new double[][] { { 0.0 }, { 0.1 }, { 0.0 }, { 0.9 } }));

        // Adding trianing instances to dataset
        Dataset trainSet = new Dataset();
        trainSet.add(trainingIns1);
        trainSet.add(trainingIns2);
        trainSet.add(trainingIns3);
        trainSet.add(trainingIns4);

        NeuralNetwork nn = new NeuralNetwork(new int[] { 3, 5, 4 }, CostFunctions.QUADRATIC,
                ActivationFunctions.SIGMOID);
        int epochs = 100;
        double learningRate = 3d;
        int miniBatchSize = 2;
        nn.stocasticGradientDescent(trainSet, epochs, learningRate, miniBatchSize);
        DoubleMatrix output = nn.feedForward(new DoubleMatrix(new double[][] { { 0.0 }, { 8.0 },
                { 0.5 } }));
        System.out.println("Prediction for input instance:");
        System.out.println(DoubleMatrixUtils.toString(output));
    }

}

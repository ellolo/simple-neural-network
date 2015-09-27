package com.penna.neural.experiments;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.jblas.DoubleMatrix;
import com.penna.neural.core.Dataset;
import com.penna.neural.core.Instance;
import com.penna.neural.core.NeuralNetwork;
import com.penna.neural.evaluation.EvaluationMetrics;
import com.penna.neural.exceptions.NetworkInitializationException;
import com.penna.neural.exceptions.NoLabelException;
import com.penna.neural.functions.ActivationFunctions;
import com.penna.neural.functions.CostFunctions;
import com.penna.neural.utils.MnistUtils;

/**
 * The goal of the MNIST experiment is to train a neural network with images of
 * digits (from 0 to 9) and then estimate the prediction accuracy on the test
 * set. Accuracy is defined as the fraction of instanced for which the network
 * predicts the correct true digit. In this experiment the parameters are not
 * optimized. Results are therefore not indicative.
 * 
 * @author mpennacchiotti
 * 
 */
public class MnistExperiment {

    public static void main(String[] argv) throws IOException, NetworkInitializationException,
            NoLabelException {
        // reading and creating training and test data
        if (argv.length < 1) {
            System.out.println("usage: MnistBinaryExperiment <path to MNIST files>");
            System.exit(-1);
        }
        String path = argv[0];
        String labelFileTr = path + "/train-labels-idx1-ubyte";
        String imageFileTr = path + "/train-images-idx3-ubyte";
        String labelFileTe = path + "/t10k-labels-idx1-ubyte";
        String imageFileTe = path + "/t10k-images-idx3-ubyte";
        Dataset trainingSet = MnistUtils.readMNISTdata(labelFileTr, imageFileTr);
        Dataset testSet = MnistUtils.readMNISTdata(labelFileTe, imageFileTe);

        // Setting up network and training
        int[] layerSizes = { 784, 30, 10 };
        NeuralNetwork nn = new NeuralNetwork(layerSizes, CostFunctions.QUADRATIC,
                ActivationFunctions.SIGMOID);
        int epochs = 20;
        double learningRate = 0.5d;
        int miniBatchSize = 10;
        nn.stocasticGradientDescent(trainingSet, epochs, learningRate, miniBatchSize);
        List<DoubleMatrix> trueLabels = new ArrayList<DoubleMatrix>();
        List<DoubleMatrix> predictedLabels = new ArrayList<DoubleMatrix>();
        for (Instance instance : testSet) {
            if (instance.isLabelled()) {
                DoubleMatrix predictedLabel = nn.feedForward(instance.getFeatures());
                predictedLabels.add(predictedLabel);
                trueLabels.add(instance.getLabels());
            }
        }
        double accuracy = EvaluationMetrics.accuracy(predictedLabels, trueLabels);
        System.out.println("Accuracy : " + String.format("%.3f", accuracy));
        double avgCosine = EvaluationMetrics.averageCosine(predictedLabels, trueLabels);
        System.out.println("Average cosine: " + String.format("%.3f", avgCosine));
    }
}
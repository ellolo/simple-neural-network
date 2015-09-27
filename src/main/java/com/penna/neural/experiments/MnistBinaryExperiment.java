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
 * The goal of the MNIST binary experiment is to train a neural network with
 * images of TWO digits (e.g. 1 and 8) and then estimate the prediction accuracy
 * on the test set. Accuracy is defined as the fraction of instanced for which
 * the network predicts the correct true digit. In this experiment the
 * parameters are not optimized. Results are therefore not indicative. This
 * experiment is a simplification of the MNIST experiment, as only two digit
 * have to be recognized.
 * 
 * @author mpennacchiotti
 * 
 */
public class MnistBinaryExperiment {

    private static Dataset getBinaryDataset(Dataset dataset, int labelOne, int labelTwo, int maxSize)
            throws NoLabelException {
        Dataset binaryDataset = new Dataset();
        int labelOneCount = 0;
        int labelTwoCount = 0;
        for (Instance instance : dataset) {
            if (instance.getLabels().get(labelOne) == 1 && labelOneCount < maxSize) {
                binaryDataset.add(instance);
                labelOneCount++;
            }
            if (instance.getLabels().get(labelTwo) == 1 && labelTwoCount < maxSize) {
                binaryDataset.add(instance);
                labelTwoCount++;
            }
            if (labelOneCount >= maxSize && labelTwoCount >= maxSize) {
                break;
            }
        }
        return binaryDataset;
    }

    /**
     * Executes the main experiment
     * 
     * @param argv no parameter accepted
     * @throws IOException
     * @throws NetworkInitializationException
     * @throws NoLabelException
     */
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
        Dataset binaryTrainingSet = getBinaryDataset(trainingSet, 1, 8, 500);
        Dataset binaryTestSet = getBinaryDataset(testSet, 1, 8, 500);

        // Setting up network and training
        int[] layerSizes = { 784, 10, 10 };
        NeuralNetwork nn = new NeuralNetwork(layerSizes, CostFunctions.QUADRATIC,
                ActivationFunctions.SIGMOID);
        int epochs = 2;
        double learningRate = 0.1d;
        int miniBatchSize = 10;
        nn.stocasticGradientDescent(binaryTrainingSet, epochs, learningRate, miniBatchSize);
        List<DoubleMatrix> trueLabels = new ArrayList<DoubleMatrix>();
        List<DoubleMatrix> predictedLabels = new ArrayList<DoubleMatrix>();
        for (Instance instance : binaryTestSet) {
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

package com.penna.neural.experiments;

import java.io.IOException;

import org.jblas.DoubleMatrix;

import com.penna.neural.core.Dataset;
import com.penna.neural.core.Instance;
import com.penna.neural.core.NeuralNetwork;
import com.penna.neural.exceptions.NetworkInitializationException;
import com.penna.neural.exceptions.NoLabelException;
import com.penna.neural.functions.ActivationFunctions;
import com.penna.neural.functions.CostFunctions;
import com.penna.neural.utils.DoubleMatrixUtils;
import com.penna.neural.utils.MnistUtils;

public class MnistExperiment {

    private static int evaluateInstanceAsClassification(NeuralNetwork nn, Instance instance) throws NoLabelException {
        // try{
        // BufferedWriter bw = new BufferedWriter(new FileWriter(new
        // File("a")));
        DoubleMatrix features = instance.getFeatures();
        /*int ct = 0;
        for (int i = 0; i < 28; i++){
            for (int j = 0; j < 28; j++){
                System.out.print(features.get(ct) + " ");
                ct++;
            }
            System.out.print("\n");
        }*/   
        int rowSize = (int) Math.sqrt(features.length);
        DoubleMatrix predictedOutput = nn.feedForward(features);
        int maxOutputIndexPred = predictedOutput.argmax();
        int maxOutputIndexGold = instance.getLabels().argmax();

        //System.out.println(DoubleMatrixUtils.toString(features));
        System.out.println(instance.getLabels());
        System.out.println(maxOutputIndexGold + " , " + maxOutputIndexPred);
        System.out.println(predictedOutput.toString());
        System.out.println("\n\n");

        if (maxOutputIndexPred == maxOutputIndexGold) {
            return 1;
        } else {
            return 0;
        }
        // }catch (IOException e){
        // System.err.println("Can't open otuput file");
        // e.printStackTrace();
        // }
        // return 0;
    }

    private static double evaluateTestSetAsClassification(NeuralNetwork nn, Dataset testSet) throws NoLabelException {
        Instance instance;
        double precision = 0d;
        for (int i = 0; i < testSet.size(); i++) {
            instance = testSet.getInstance(i);
            if (i % 50 == 0) {
                System.out.println("\tTesting instances (" + i + " of " + testSet.size() + ")");
            }
            precision += evaluateInstanceAsClassification(nn, instance);
        }
        precision /= testSet.size();
        return precision;
    }

    public static void experiment(Dataset trainingSet, Dataset testSet, int[] layerSizes,
            int epochs, double eta, int miniBatchSize) throws NetworkInitializationException, NoLabelException {
        NeuralNetwork nn = new NeuralNetwork(layerSizes, CostFunctions.QUADRATIC,
                ActivationFunctions.SIGMOID);
        System.out.println("\n*** TRIANING PHASE ***");
        // System.out.println((nn.getWeights()[1]));
        nn.stocasticGradientDescent(trainingSet, epochs, eta, miniBatchSize);
        System.out.println("\n*** FINAL NN ***");
        System.out.println("\n*** Biases");
        for (DoubleMatrix d :  nn.getBiases()){
            System.out.println(DoubleMatrixUtils.toString(d));
        }
        System.out.println("\n*** weights");
        for (DoubleMatrix d :  nn.getWeights()){
            System.out.println(DoubleMatrixUtils.toString(d));
        }
        System.out.println("\n*** TEST PHASE ***");
        double accuracy = evaluateTestSetAsClassification(nn, testSet);
        System.out.println("Test Accuracy : " + accuracy);
    }

    public static void main(String[] argv) throws IOException, NetworkInitializationException, NoLabelException {
        //int[] layerSizes = { 784, 30, 10 };
        int[] layerSizes = { 784, 30, 10 };
        // GoldSet trainingSet = new GoldSet(layerSizes[0],
        // layerSizes[layerSizes.length-1], 1134);
        // GoldSet testSet = new GoldSet(layerSizes[0],
        // layerSizes[layerSizes.length-1], 100);

        String labelFileTr = "/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist/training/train-labels-idx1-ubyte";
        String imageFileTr = "/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist/training/train-images-idx3-ubyte";
        String labelFileTe = "/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist/test/t10k-labels-idx1-ubyte";
        String imageFileTe = "/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist/test/t10k-images-idx3-ubyte";
        Dataset trainingSet = MnistUtils.readMNISTdata(labelFileTr, imageFileTr);
        Dataset testSet = MnistUtils.readMNISTdata(labelFileTe, imageFileTe);
        trainingSet = trainingSet.getSubSet(0, 3000);
        experiment(trainingSet, testSet, layerSizes, 30, 0.1, 10);
        //experiment(trainingSet, testSet, layerSizes,50, 1, 50);
        /*
        Dataset tr = new Dataset();
        int ones = 0;
        int eights = 0;
        int mx = 1000;
        for (int i = 0 ; i < trainingSet.size(); i++){
            Instance ins = trainingSet.getInstance(i);
            if (ins.getLabels().get(1) == 1 && ones < mx){
               tr.add(ins);
               ones++;
            }
            if (ins.getLabels().get(8) == 1 && eights < mx){
                tr.add(ins);
                eights++;
             }
            if (ones >= mx && eights >=mx){
                break;
            }
        }
        Dataset te = new Dataset();
        ones = 0;
        eights = 0;
        mx = 30;
        for (int i = 0 ; i < testSet.size(); i++){
            Instance ins = trainingSet.getInstance(i);
            if (ins.getLabels().get(1) == 1 && ones < mx){
               te.add(ins);
               ones++;
            }
            if (ins.getLabels().get(8) == 1 && eights < mx){
                te.add(ins);
                eights++;
             }
            if (ones >= mx && eights >=mx){
                break;
            }
        }
       
        experiment(tr, te, layerSizes, 30, 0.1, 10);
        experiment(tr, te, layerSizes, 30, 0.1, 10);
        experiment(tr, te, layerSizes, 30, 0.1, 10);*/
    }

}

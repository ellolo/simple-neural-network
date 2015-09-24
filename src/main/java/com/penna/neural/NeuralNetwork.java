package com.penna.neural;

import java.util.logging.Level;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;
import com.penna.neural.exceptions.NetworkInitializationException;
import com.penna.neural.common.*;
import com.penna.neural.functions.*;

public class NeuralNetwork {

    private static final Logger LOGGER = Logger.getLogger(DatasetReader.class.getName());

    private final int totLayer;
    private final int[] layerSizes;
    // biases are array of vectors
    private DoubleMatrix[] biases;
    // weights array of matrixes
    private DoubleMatrix[] weights;
    private CostFunctions costFunction;
    private ActivationFunctions activationFunction;

    private void randomInitialization() {
        biases = new DoubleMatrix[totLayer - 1];
        weights = new DoubleMatrix[totLayer - 1];
        for (int layer = 1; layer <= biases.length; layer++) {
            int layerSize = layerSizes[layer];
            int prevLayerSize = layerSizes[layer - 1];
            biases[layer - 1] = DoubleMatrix.randn(layerSize, 1);
            weights[layer - 1] = DoubleMatrix.randn(layerSize, prevLayerSize);
            LOGGER.fine("layer " + layer + " initialization value :" + "\nbiases\n"
                    + DoubleMatrixUtils.toString(biases[layer - 1]) + "\nweights\n"
                    + DoubleMatrixUtils.toString(weights[layer - 1]));
        }
    }

    private void validateLayers(int[] layerSizes) throws NetworkInitializationException {
        for (int size : layerSizes) {
            if (size < 1) {
                throw new NetworkInitializationException("Layer size must be bigger than 0");
            }
        }
    }

    private void validateParameters(int[] layerSizes, DoubleMatrix[] biases, DoubleMatrix[] weights)
            throws NetworkInitializationException {
        if (biases.length != layerSizes.length - 1 || weights.length != layerSizes.length - 1) {
            throw new NetworkInitializationException("Number of biases and weights must equal the "
                    + "number of layers");
        }
        for (int i = 1; i < layerSizes.length; i++) {
            if (biases[i - 1].rows != layerSizes[i] || biases[i - 1].columns != 1
                    || weights[i - 1].rows != layerSizes[i]
                    || weights[i - 1].columns != layerSizes[i - 1]) {
                throw new NetworkInitializationException("Bias and weight values for layer " + i
                        + " must match the number of neurons");
            }
        }
    }

    // TODO: check costFunc and neuron and emit exception if needed
    public NeuralNetwork(int[] layerSizes, CostFunctions costFunc, ActivationFunctions neuron)
            throws NetworkInitializationException {
        validateLayers(layerSizes);
        this.layerSizes = layerSizes;
        this.totLayer = this.layerSizes.length;
        this.costFunction = costFunc;
        this.activationFunction = neuron;
        randomInitialization();
        LOGGER.info("Random initialization completed successfully");
    }

    public NeuralNetwork(int[] layerSizes, DoubleMatrix[] biases, DoubleMatrix[] weights,
            CostFunctions costFunc, ActivationFunctions neuron)
            throws NetworkInitializationException {
        validateLayers(layerSizes);
        validateParameters(layerSizes, biases, weights);
        this.layerSizes = layerSizes;
        this.totLayer = this.layerSizes.length;
        this.biases = biases.clone();
        this.weights = weights.clone();
        this.costFunction = costFunc;
        this.activationFunction = neuron;
        LOGGER.info("Initialization completed successfully");
    }

    public DoubleMatrix[] getWeights() {
        return weights;
    }

    public DoubleMatrix[] getBiases() {
        return biases;
    }

    private DoubleMatrix[] initializeDeltaMatrixes(boolean isBias) {
        DoubleMatrix[] deltaMatrix = new DoubleMatrix[totLayer - 1];
        for (int layer = 1; layer <= biases.length; layer++) {
            int layerSize = layerSizes[layer];
            if (isBias) {
                deltaMatrix[layer - 1] = DoubleMatrix.zeros(layerSize, 1);
            } else {
                int prevLayerSize = layerSizes[layer - 1];
                deltaMatrix[layer - 1] = DoubleMatrix.zeros(layerSize, prevLayerSize);
            }
        }
        return deltaMatrix;
    }

    private DoubleMatrix[] initializeActivationMatrixes() {
        DoubleMatrix[] activations = new DoubleMatrix[totLayer];
        for (int layer = 0; layer < totLayer; layer++) {
            int layerSize = layerSizes[layer];
            activations[layer] = DoubleMatrix.zeros(layerSize, 1);
        }
        return activations;
    }

    private ParameterDeltas backPropagation(Instance instance) {
        DoubleMatrix[] deltaBiases = initializeDeltaMatrixes(true);
        DoubleMatrix[] deltaWeights = initializeDeltaMatrixes(false);
        DoubleMatrix[] activations = initializeActivationMatrixes();
        DoubleMatrix delta;
        // feed-forward
        // fist layer activations are the actual inputs
        //System.out.println("***INSTANCE*** " + instance.getFeatures());
        activations[0] = instance.getFeatures();
        //System.out.println(" activation 0 : \n" + DoubleMatrixUtils.toString(activations[0]));
        for (int layer = 1; layer < totLayer; layer++) {
            //System.out.println(" weights : " + DoubleMatrixUtils.toString(weights[layer - 1]));
            //System.out.println(" biases : " + DoubleMatrixUtils.toString(biases[layer - 1]));
            DoubleMatrix zetas = (weights[layer - 1].mmul(activations[layer - 1])).add(biases[layer - 1]);
            //System.out.println(" zetas : " + DoubleMatrixUtils.toString(zetas));
            activations[layer] = activationFunction.activate(zetas);
            //System.out.println(" activation " + layer + " : \n" + DoubleMatrixUtils.toString(activations[layer]));
        }
        // backward-propagation
        //delta = deltaOutLayer(activations[totLayer-1], instance.getLabels());
        //System.out.println("**Backtrack " + instance.getFeatures());
        //System.out.println(" activation " + (totLayer-1)+ " : \n" + DoubleMatrixUtils.toString(activations[totLayer-1]));
        delta = costFunction.derivative(activations[totLayer - 1], instance.getLabels(), activationFunction); // a @ (1-a) @ (-(y-a))
        //System.out.println(" delta : \n" + DoubleMatrixUtils.toString(delta));
         // delta * a_prev
        deltaWeights[totLayer - 2] = delta.mmul(activations[totLayer - 2].transpose()); 
        deltaBiases[totLayer - 2] = delta;
        //System.out.println(" delta biases: \n" + DoubleMatrixUtils.toString(deltaBiases[totLayer - 2]));
        for (int layer = totLayer - 2; layer > 0; layer--) {
             //delta = deltaHiddenLayer(activations[layer], weights[layer], delta);
             // a @ (1-a) @ (next_w * next_delta)
            DoubleMatrix activationDeriv = activationFunction.derivative(activations[layer]);
            delta = (weights[layer].transpose().mmul(delta)).mul(activationDeriv); 
            deltaWeights[layer - 1] = delta.mmul(activations[layer - 1].transpose());
            deltaBiases[layer - 1] = delta;
        }
        ParameterDeltas parameterDeltas = new ParameterDeltas(deltaWeights, deltaBiases);
        return parameterDeltas; // <>{deltaBiases, deltaWeights}
    }

    
    private DoubleMatrix deltaHiddenLayer(DoubleMatrix activations, DoubleMatrix nextLyrWeights,
            DoubleMatrix nextLyrDelta) {
        DoubleMatrix sigm = sigmoidDerivative(activations);
        DoubleMatrix costDeriv = nextLyrWeights.transpose().mmul(nextLyrDelta);
        return sigm.mul(costDeriv); // a @ (1-a) @ (next_w * next_delta) }
    }

    private DoubleMatrix deltaOutLayer(DoubleMatrix activations, DoubleMatrix output) {
        DoubleMatrix sigm = sigmoidDerivative(activations);
        DoubleMatrix costDeriv = quadraticCostOutputDerivative(activations, output);
        return sigm.mul(costDeriv); // a @ (1-a) @ (-(y-a)) }
    }

    private DoubleMatrix sigmoidDerivative(DoubleMatrix activations) {
        return activations.mul(DoubleMatrix.ones(activations.rows, 1).sub(activations));
        // a @ (1-a) }
    }

    private DoubleMatrix quadraticCostOutputDerivative(DoubleMatrix activations, DoubleMatrix output) {
        return activations.sub(output); // -(y-a)
    }
     
    private void updateParameters(Dataset trainingBatch, double learnRate, int batchSize) {
        //for (Instance instance: trainingBatch){
        for (int i = 0; i < trainingBatch.size(); i++){
            Instance instance = trainingBatch.getInstance(i);
            ParameterDeltas parameterDeltas = backPropagation(instance);
            DoubleMatrix[] deltaBiases = parameterDeltas.deltaBiases;
            DoubleMatrix[] deltaWeights = parameterDeltas.deltaWeights;
            for (int layer = 1; layer < totLayer; layer++) {
                biases[layer - 1] = biases[layer - 1].sub((deltaBiases[layer - 1]).mul(learnRate).div(batchSize));
                weights[layer - 1] = weights[layer - 1].sub((deltaWeights[layer - 1]).mul(learnRate).div(batchSize));
            }    
        }
        
    }

    public void stocasticGradientDescent(Dataset trainingSet, int numEpochs, double learnRate,
            int miniBatchSize) {
        Dataset trainingBatch;
        int numBatches = trainingSet.size() / miniBatchSize; // TODO should add
                                                             // one more bucket
        int currIndex;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainingSet.shuffle();
            currIndex = 0;
            System.out.println("\nEpoch : " + epoch);
            for (int j = 0; j < numBatches - 1; j++) {
                trainingBatch = trainingSet.getSubSet(currIndex, currIndex + miniBatchSize);
                //System.out.println("\tEpoch : " + epoch + " minibatch : " + j + " (sublist from "
                //        + currIndex + " to " + (currIndex + miniBatchSize) + " : "
                //        + trainingBatch.size() + ")");
                updateParameters(trainingBatch, learnRate, trainingBatch.size());
                currIndex = currIndex + miniBatchSize;
            }
            trainingBatch = trainingSet.getSubSet(currIndex, trainingSet.size());
            //System.out.println("\tEpoch : " + epoch + " minibatch : " + (numBatches - 1)
            //        + " (sublist from " + currIndex + " to " + (currIndex + miniBatchSize) + " : "
            //       + trainingBatch.size() + ")");
            updateParameters(trainingBatch, learnRate, trainingBatch.size());
        }
    }

    /**
     * Performs feedforward on a given user input. z_i = sum( w_i * a_{i-1} +
     * b_i) a_i = activationFunc(z_i)
     * 
     * @param input
     *            input layer for which to evaluate the output
     * @return neural network output
     */
    public DoubleMatrix feedForward(DoubleMatrix input) {
        DoubleMatrix layerOutput = input;
        for (int layer = 1; layer <= biases.length; layer++) {
            DoubleMatrix zetas = (weights[layer - 1].mmul(layerOutput)).add(biases[layer - 1]);
            layerOutput = activationFunction.activate(zetas);
        }
        return layerOutput;
    }

    public static void main(String[] argv) throws NetworkInitializationException {

        /*** TESTING FEEDFORWARD ***/
        DoubleMatrix weightsLayer1 = new DoubleMatrix(new double[][] { { 4.0d, 3.0d, -2.5d },
                { 1.5d, 0.0d, -1.0d } });
        DoubleMatrix weightsLayer2 = new DoubleMatrix(new double[][] { { 1.0d, -2.5d },
                { 2.0d, 3.5d }, { 0.0d, 1.0d }, { -2.0d, 1.5d } });
        DoubleMatrix weightsLayer3 = new DoubleMatrix(new double[][] {
                { 1.0d, -2.5d, -1.0d, 2.0d }, { 2.0d, 3.5d, -2.0d, 1.5d } });

        DoubleMatrix biasesLayer1 = new DoubleMatrix(new double[][] { { 1.0 }, { -0.5 } });
        DoubleMatrix biasesLayer2 = new DoubleMatrix(new double[][] { { -3.0 }, { 1.5 }, { 0.0 }, { -1.5 } });
        //DoubleMatrix biasesLayer3 = new DoubleMatrix(new double[][] { { -1.0 }, { 2.0 } });
        //DoubleMatrix[] weights = { weightsLayer1, weightsLayer2, weightsLayer3 };
        DoubleMatrix[] weights = { weightsLayer1, weightsLayer2};
        //DoubleMatrix[] biases = { biasesLayer1, biasesLayer2, biasesLayer3 };
        DoubleMatrix[] biases = { biasesLayer1, biasesLayer2};
        //NeuralNetwork nnTest = new NeuralNetwork(new int[]{3,2,4,2}, biases, weights, CostFunctions.QUADRATIC, ActivationFunctions.SIGMOID);
        //NeuralNetwork nnTest = new NeuralNetwork(new int[]{3,2,4}, biases, weights, CostFunctions.QUADRATIC, ActivationFunctions.SIGMOID);
        NeuralNetwork nnTest = new NeuralNetwork(new int[]{3,2,4}, CostFunctions.QUADRATIC, ActivationFunctions.SIGMOID);
        Instance i1 = new Instance(new DoubleMatrix(new double[][] { { 1.0 }, { -0.5 }, { 3.0 } }), new DoubleMatrix(new double[][] { { 0.1 }, { 0.8 }, { 0.05 }, { 0.05 } }));
        Instance i2 = new Instance(new DoubleMatrix(new double[][] { { 0.0 }, { 7.0 }, { -0.5 } }), new DoubleMatrix(new double[][] { { 0.0 }, { 0.0 }, { 0.1 }, { 0.9 } }));
        Instance i3 = new Instance(new DoubleMatrix(new double[][] { { 1.5 }, { -1.0 }, { 5.0 } }), new DoubleMatrix(new double[][] { { 0.0 }, { 0.9 }, { 0.1 }, { 0.0 } }));
        Instance i4 = new Instance(new DoubleMatrix(new double[][] { { 0.0 }, { 6.0 }, { 0.0 } }), new DoubleMatrix(new double[][] { { 0.0 }, { 0.1 }, { 0.0 }, { 0.9 } }));
        System.out.println("BETA 0" + DoubleMatrixUtils.toString(nnTest.biases[0]));
        
        Dataset d = new Dataset();
        d.add(i1);
        d.add(i2);
        d.add(i3);
        d.add(i4);
        nnTest.stocasticGradientDescent(d, 20, 3, 2);
        DoubleMatrix output = nnTest.feedForward(new DoubleMatrix(new double[][]{{ 10.0 }, { 8.0 }, { 0.5 }}));
        System.out.println("OUTPUT");
        System.out.println(DoubleMatrixUtils.toString(output));
        
        //DoubleMatrix weightsLayer1 = new DoubleMatrix(new double[][] { { 4.0d, 3.0d, -2.5d },
        //        { 1.5d, 0.0d, -1.0d } });
        //DoubleMatrix weightsLayer2 = new DoubleMatrix(new double[][] { { 1.0d, -2.5d },
        //        { 2.0d, 3.5d }, { 0.0d, 1.0d }, { -2.0d, 1.5d } });
        //DoubleMatrix weightsLayer3 = new DoubleMatrix(new double[][] {
         //       { 1.0d, -2.5d, -1.0d, 2.0d }, { 2.0d, 3.5d, -2.0d, 1.5d } });
        //DoubleMatrix biasesLayer1 = new DoubleMatrix(new double[][] { { 1.0 }, { -0.5 } });
        //DoubleMatrix biasesLayer2 = new DoubleMatrix(new double[][] { { -3.0 }, { 1.5 }, { 0.0 }, { -1.5 } });
        //DoubleMatrix biasesLayer3 = new DoubleMatrix(new double[][] { { -1.0 }, { 2.0 } });
        //DoubleMatrix[] weights = { weightsLayer1, weightsLayer2, weightsLayer3 };
        //DoubleMatrix[] biases = { biasesLayer1, biasesLayer2, biasesLayer3 };
        //NeuralNetwork nnTest = new NeuralNetwork(new int[]{3,2,4,2}, biases, weights, CostFunctions.QUADRATIC, ActivationFunctions.SIGMOID);  
        //DoubleMatrix output = nnTest.feedForward(new DoubleMatrix(new double[][]{{2.0d},{-1.5d},{1.0d}}));
         //System.out.println(output.toString("%.3f"));

        // NeuralNetwork nnTest = new NeuralNetwork(new int[]{100, 50, 45});
        // DoubleMatrix output = nnTest.feedForward(DoubleMatrix.randn(100,1));
        // System.out.println(output.toString("%.3f"));
        // nnTest.stocasticGradientDescent(new GoldSet(100,45, 52), 4, 0.8, 10);
    }

}

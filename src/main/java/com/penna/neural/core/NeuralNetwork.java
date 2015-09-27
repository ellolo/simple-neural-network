package com.penna.neural.core;

import java.util.logging.Logger;
import org.jblas.DoubleMatrix;
import com.penna.neural.exceptions.NetworkInitializationException;
import com.penna.neural.exceptions.NoLabelException;
import com.penna.neural.functions.*;
import com.penna.neural.utils.*;

/**
 * Implementation of a simple neural network. Activations, weights and biases are represented with
 * jblas DoubleMatrixes. The neural network supports the following functions:
 * Cost functions:
 * - quadratic
 * - cross entropy
 * Activation functions:
 * - sigmoid
 * - tanh
 * Training is performed by gradient descent/backtracking.
 *  
 * @author mpennacchiotti
 *
 */
public class NeuralNetwork {

    private static final Logger LOGGER = Logger.getLogger(MnistUtils.class.getName());
    private static final int MAX_LAYER_SIZE = 10000;
    private static final int MAX_NUM_LAYERS = 10;

    private final int totLayer;
    // number of neurons for each layer
    private final int[] layerSizes;
    // biases, array of one dimensional matrix
    private DoubleMatrix[] biases;
    // weights,  array of matrixes
    private DoubleMatrix[] weights;
    // cost function used by the network
    private CostFunctions costFunction;
    // activation function used by the network
    private ActivationFunctions activationFunction;
  

    /**
     * Construct a neural network, given the specifics in input. All parameters are initialized
     * randomly. This is the preferred network constructor 
     * 
     * @param layerSizes an array containing the number of neurons for each layer
     * @param costFunc the cost function that will be use by the netowrk 
     * @param activFunc the activation function that will be use by the netowrk
     * @throws NetworkInitializationException
     */
    public NeuralNetwork(int[] layerSizes, CostFunctions costFunc, ActivationFunctions activFunc)
            throws NetworkInitializationException {
        validateLayers(layerSizes);
        this.layerSizes = layerSizes;
        this.totLayer = this.layerSizes.length;
        this.costFunction = costFunc;
        this.activationFunction = activFunc;
        randomInitialization();
        LOGGER.info("Random initialization completed successfully");
    }
    
    /**
     * Construct a neural network, given the specifics in input. Parameters values are input
     * explicitly
     * 
     * @param layerSizes an array containing the number of neurons for each layer
     * @param biases an array containing the biases' matrix of each layer 
     * @param weightsan array containing the weights' matrix of each layer
     * @param costFunc the cost function that will be use by the netowrk 
     * @param activFunc the activation function that will be use by the netowrk
     * @throws NetworkInitializationException
     */
    public NeuralNetwork(int[] layerSizes, DoubleMatrix[] biases, DoubleMatrix[] weights,
            CostFunctions costFunc, ActivationFunctions activFunc)
            throws NetworkInitializationException {
        validateLayers(layerSizes);
        validateParameters(layerSizes, biases, weights);
        this.layerSizes = layerSizes;
        this.totLayer = this.layerSizes.length;
        this.biases = biases.clone();
        this.weights = weights.clone();
        this.costFunction = costFunc;
        this.activationFunction = activFunc;
        LOGGER.info("Initialization completed successfully");
    }
    
    /**
     * Randomly initializes the parameters of the neural network using from a normal
     * distribution. 
     */
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

    /**
     * Validates that, for each layer, there is at least one neuron 
     * 
     * @param layerSizes an array containing the number of neurons for each layer
     * @throws NetworkInitializationException if validation fails
     */
    private void validateLayers(int[] layerSizes) throws NetworkInitializationException {
        if (layerSizes.length < 2 || layerSizes.length > MAX_NUM_LAYERS){
            throw new NetworkInitializationException("Number of layers must be between 2 and " + MAX_NUM_LAYERS);
        }
        for (int size : layerSizes) {
            if (size < 1 || size > MAX_LAYER_SIZE) {
                throw new NetworkInitializationException("Layer size must be between 0 and " + MAX_LAYER_SIZE);
            }
        }
    }

    /**
     * Validates that the biases and the weights conform to the specified layer sizes.
     * 
     * @param layerSizes an array containing the number of neurons for each layer 
     * @param biases an array containing the biases' matrix of each layer
     * @param weights an array containing the weights' matrix of each layer
     * @throws NetworkInitializationException
     */
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
    
    public DoubleMatrix[] getWeights() {
        return weights;
    }

    public DoubleMatrix[] getBiases() {
        return biases;
    }
    
    /**
     * Initializes delta parameter matrixes, weights and biases, to zero for every layer. These
     * are the matrixes that store the output of the backtracking algorithm.
     * 
     * @return initialized weights and bias matrixes
     */
    private ParameterDeltas initializeDeltaParameters() {
        DoubleMatrix[] weightMatrix = new DoubleMatrix[totLayer - 1];
        DoubleMatrix[] biasMatrix = new DoubleMatrix[totLayer - 1];
        for (int layer = 1; layer <= biases.length; layer++) {
            int layerSize = layerSizes[layer];
            int prevLayerSize = layerSizes[layer - 1];
            weightMatrix[layer - 1] = DoubleMatrix.zeros(layerSize, prevLayerSize);
            biasMatrix[layer - 1] = DoubleMatrix.zeros(layerSize, 1);
        }
        return new ParameterDeltas(weightMatrix, biasMatrix);
    }

    /**
     * Initializes activation matrix to zero for every layer. This matrix is used by
     * backpropagation during the learning phase. 
     * 
     * @return initialized activation matrix
     */
    private DoubleMatrix[] initializeActivation() {
        DoubleMatrix[] activations = new DoubleMatrix[totLayer];
        for (int layer = 0; layer < totLayer; layer++) {
            int layerSize = layerSizes[layer];
            activations[layer] = DoubleMatrix.zeros(layerSize, 1);
        }
        return activations;
    }

    /**
     * Executes backpropagation for a single instance.
     * 
     * @param instance the instance for which to backpropagate
     * @return parameter delta, weights and biases
     * @throws NoLabelException if the instance doesn't have a label, i.e. is not a trining intance
     */
    private ParameterDeltas backPropagation(Instance instance) throws NoLabelException {
        // Notation:
        // w = weights at layer
        // b = biases at layer
        // a = activations at layer
        // z = zeta, i.e. w * a-1 + b
        // d = delta, i.e. the derivative of the cost function  w.r.t. zeta
        // -1 = previous layer
        ParameterDeltas initializedDeltas = initializeDeltaParameters();  
        DoubleMatrix[] deltaWeights = initializedDeltas.deltaWeights;
        DoubleMatrix[] deltaBiases = initializedDeltas.deltaBiases;
        DoubleMatrix[] activations = initializeActivation();
        // feed forward
        activations[0] = instance.getFeatures();
        for (int layer = 1; layer < totLayer; layer++) {
            // z = w * a_-1 + b 
            DoubleMatrix zetas = (weights[layer - 1].mmul(activations[layer - 1])).add(biases[layer - 1]);
            activations[layer] = activationFunction.activate(zetas);
        }
        // backward propagation
        // d = a @ (1-a) @ (-(y-a))
        DoubleMatrix delta = costFunction.derivative(activations[totLayer - 1], instance.getLabels(), activationFunction); 
        LOGGER.fine(" delta: \n" + DoubleMatrixUtils.toString(deltaBiases[totLayer - 2]));
        // w = d * a_-1
        deltaWeights[totLayer - 2] = delta.mmul(activations[totLayer - 2].transpose()); 
        // b = d
        deltaBiases[totLayer - 2] = delta;
        for (int layer = totLayer - 2; layer > 0; layer--) {
            DoubleMatrix activationDeriv = activationFunction.derivative(activations[layer]);
            // d = a @ (1-a) @ (w_+1 * d_+1)
            delta = (weights[layer].transpose().mmul(delta)).mul(activationDeriv); 
            // w = d * a_-1
            deltaWeights[layer - 1] = delta.mmul(activations[layer - 1].transpose());
            // b = d
            deltaBiases[layer - 1] = delta;
        }
        ParameterDeltas parameterDeltas = new ParameterDeltas(deltaWeights, deltaBiases);
            return parameterDeltas;
    }

    /**
     * Performs backpropagation for the instances in the input dataset, and updates the weights and
     * biases according to the increment matrixes returned by the backpropagation.
     *  
     * @param trainingBatch the set of instances to be backpropagated
     * @param learnRate the learning rate for the increments
     * @param batchSize size of the set of instances
     */
    private void updateParameters(Dataset trainingBatch, double learnRate, int batchSize) {
        //for (Instance instance: trainingBatch){
        for (int i = 0; i < trainingBatch.size(); i++){
            try{
                Instance instance = trainingBatch.getInstance(i);
                ParameterDeltas parameterDeltas = backPropagation(instance);
                DoubleMatrix[] deltaBiases = parameterDeltas.deltaBiases;
                DoubleMatrix[] deltaWeights = parameterDeltas.deltaWeights;
                for (int layer = 1; layer < totLayer; layer++) {
                    biases[layer - 1] = biases[layer - 1].sub((deltaBiases[layer - 1]).mul(learnRate).div(batchSize));
                    weights[layer - 1] = weights[layer - 1].sub((deltaWeights[layer - 1]).mul(learnRate).div(batchSize));
                }
            } catch (NoLabelException nle) {
                LOGGER.severe("Found instance without label. Learning may be unstable");
            }
        }
    }

    /**
     * Performs gradient descent learning on a given dataset.
     * 
     * @param trainingSet the training set used for learning
     * @param numEpochs the number of epochs of the training 
     * @param learnRate the learning rate for parameter updates
     * @param miniBatchSize minibatch size for parameter update
     */
    public void stocasticGradientDescent(Dataset trainingSet, int numEpochs, double learnRate,
            int miniBatchSize) throws NetworkInitializationException {
        if (trainingSet.size() < 1 || numEpochs < 1 || learnRate <= 0 || miniBatchSize < 1){
            throw new NetworkInitializationException("Invalid gradient descent parameters.");
        }
        Dataset trainingBatch;
        int numBatches = trainingSet.size() / miniBatchSize;
        int currIndex;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainingSet.shuffle();
            currIndex = 0;
            LOGGER.info("Gradient descent epoch : " + epoch);
            for (int j = 0; j < numBatches - 1; j++) {
                trainingBatch = trainingSet.getSubSet(currIndex, currIndex + miniBatchSize);
                LOGGER.fine("  Minibatch: " + j);
                updateParameters(trainingBatch, learnRate, trainingBatch.size());
                currIndex = currIndex + miniBatchSize;
            }
            trainingBatch = trainingSet.getSubSet(currIndex, trainingSet.size());
            LOGGER.fine("  Minibatch: " + numBatches);
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
}

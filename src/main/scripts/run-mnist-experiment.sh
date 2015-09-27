#Runs the mnist full neural network experiment

MNIST_PATH=/Users/mpennacchiotti/dev/deep-learning/neural-net/data/mnist

java -Xmx2g -cp  ../../../target/neural-0.1-jar-with-dependencies.jar com.penna.neural.experiments.MnistExperiment $MNIST_PATH

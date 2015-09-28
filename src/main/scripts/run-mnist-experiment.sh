#Runs the mnist full neural network experiment

MNIST_PATH=../../../data/mnist/

java -Xmx2g -cp  ../../../target/neural-0.1-jar-with-dependencies.jar com.penna.neural.experiments.MnistExperiment $MNIST_PATH

Neural Network
==============

This project implements a simple neural network with the following features:
  - learning via backpropagation
  - cost functions: quadratic / cross-entropy
  - activation function: sigmoid / tanh

The project also contain examples of experiments that use the neural network:
please refer to source code in `com.penna.neural.experiments` for more information.

Compiling
---------
Install Maven and then execute on the project root

    $ mvn clean package

Running sample experiment
-------------------------
From project root:

    $ cd src/main/scripts
    $ sh <experiment runner>

where `<experiment runner>` is a shell script that executes one of the
experiments in the `script` directory. 

Further information and questions
---------------------------------
email to <marco.pennacchiotti@gmail.com>

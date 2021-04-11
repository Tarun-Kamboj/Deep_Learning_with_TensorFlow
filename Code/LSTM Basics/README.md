# Long Short-Term Memory (LSTM) Model

![](http://ForTheBadge.com/images/badges/made-with-python.svg) 

## Dependencies

![](https://img.shields.io/badge/TensorFlow-2.1.0-FF6F00?style=for-the-badge&logo=TensorFlow)
![](https://img.shields.io/badge/numpy-1.19.2-013243?style=for-the-badge&logo=NumPy)

## Introduction

Recurrent Neural Networks are Deep Learning models with simple structures and a feedback mechanism built-in, or in different words, the output of a layer is added to the next input and fed back to the same layer.

The Recurrent Neural Network is a specialized type of Neural Network that solves the issue of **maintaining context for Sequential data** -- such as Weather data, Stocks, Genes, etc. At each iterative step, the processing unit takes in an input and the current state of the network, and produces an output and a new state that is <b>re-fed into the network</b>.

<p align="center">
	<img width="500px" src="Images/img0.png">
	Representation of a Recurrent Neural Network
</p>

However, <b>this model has some problems</b>. It's very computationally expensive to maintain the state for a large amount of units, even more so over a long amount of time. Additionally, Recurrent Networks are very sensitive to changes in their parameters. As such, they are prone to different problems with their Gradient Descent optimizer -- they either grow exponentially (Exploding Gradient) or drop down to near zero and stabilize (Vanishing Gradient), both problems that greatly harm a model's learning capability.

To solve these problems, Hochreiter and Schmidhuber published a paper in 1997 describing a way to keep information over long periods of time and additionally solve the oversensitivity to parameter changes, i.e., make backpropagating through the Recurrent Networks more viable. This proposed method is called Long Short-Term Memory (LSTM).

## Long Short-Term Memory Model

The Long Short-Term Memory, as it was called, was an abstraction of how computer memory works. It is "bundled" with whatever processing unit is implemented in the Recurrent Network, although outside of its flow, and is responsible for keeping, reading, and outputting information for the model. The way it works is simple: you have a linear unit, which is the information cell itself, surrounded by three logistic gates responsible for maintaining the data. One gate is for inputting data into the information cell, one is for outputting data from the input cell, and the last one is to keep or forget data depending on the needs of the network.

Thanks to that, it not only solves the problem of keeping states, because the network can choose to forget data whenever information is not needed, it also solves the gradient problems, since the Logistic Gates have a very nice derivative.

<h3>Long Short-Term Memory Architecture</h3>

The Long Short-Term Memory is composed of a linear unit surrounded by three logistic gates. The name for these gates vary from place to place, but the most usual names for them are:

<ul>
    <li>the "Input" or "Write" Gate, which handles the writing of data into the information cell</li>
    <li>the "Output" or "Read" Gate, which handles the sending of data back onto the Recurrent Network</li>
    <li>the "Keep" or "Forget" Gate, which handles the maintaining and modification of the data stored in the information cell</li>
</ul>

<p align="center">
	<img width="500px" src="Images/img1.png">
	Diagram of the Long Short-Term Memory Unit
</p>

The three gates are the centerpiece of the LSTM unit. The gates, when activated by the network, perform their respective functions. For example, the Input Gate will write whatever data it is passed into the information cell, the Output Gate will return whatever data is in the information cell, and the Keep Gate will maintain the data in the information cell. These gates are analog and multiplicative, and as such, can modify the data based on the signal they are sent.

For example, an usual flow of operations for the LSTM unit is as such: First off, the Keep Gate has to decide whether to keep or forget the data currently stored in memory. It receives both the input and the state of the Recurrent Network, and passes it through its Sigmoid activation. If K<sub>t</sub> has value of 1 means that the LSTM unit should keep the data stored perfectly and if K<sub>t</sub> a value of 0 means that it should forget it entirely. Consider S<sub>t-1</sub> as the incoming (previous) state, x<sub>t</sub> as the incoming input, and W<sub>k</sub>, B<sub>k</sub> as the weight and bias for the Keep Gate. Additionally, consider Old<sub>t-1</sub> as the data previously in memory. What happens can be summarized by this equation:

<strong>
K<sub>t</sub> = σ(W<sub>k</sub> × [S<sub>t-1</sub>, x<sub>t</sub>] + B<sub>k</sub>)

Old<sub>t-1</sub> = K<sub>t</sub> × Old<sub>t-1</sub>
</strong>

As you can see, Old<sub>t-1</sub> was multiplied by value was returned by the Keep Gate(K<sub>t</sub>) -- this value is written in the memory cell.

<br>
Then, the input and state are passed on to the Input Gate, in which there is another Sigmoid activation applied. Concurrently, the input is processed as normal by whatever processing unit is implemented in the network, and then multiplied by the Sigmoid activation's result I<sub>t</sub>, much like the Keep Gate. Consider W<sub>i</sub> and B<sub>i</sub> as the weight and bias for the Input Gate, and C<sub>t</sub> the result of the processing of the inputs by the Recurrent Network.
<br><br>

<strong>
I<sub>t</sub> = σ(W<sub>i</sub> × [S<sub>t-1</sub>,x<sub>t</sub>]+B<sub>i</sub>)

New<sub>t</sub> = I<sub>t</sub> × C<sub>t</sub>
</strong>

New<sub>t</sub> is the new data to be input into the memory cell. This is then <b>added</b> to whatever value is still stored in memory.

<strong>
Cell<sub>t</sub> = Old<sub>t</sub> + New<sub>t</sub>
</strong>

<br>
We now have the <i>candidate data</i> which is to be kept in the memory cell. The conjunction of the Keep and Input gates work in an analog manner, making it so that it is possible to keep part of the old data and add only part of the new data. Consider however, what would happen if the Forget Gate was set to 0 and the Input Gate was set to 1:

<strong>
Old<sub>t</sub> = 0 × Old<sub>t-1</sub>

New<sub>t</sub> = 1 × C<sub>t</sub>

Cell<sub>t</sub> = C<sub>t</sub>
</strong></font>

<br>
The old data would be totally forgotten and the new data would overwrite it completely.

The Output Gate functions in a similar manner. To decide what we should output, we take the input data and state and pass it through a Sigmoid function as usual. The contents of our memory cell, however, are pushed onto a <i>Tanh</i> function to bind them between a value of -1 to 1. Consider W<sub>o</sub> and B<sub>o</sub> as the weight and bias for the Output Gate.

<strong>
O<sub>t</sub> = σ(W<sub>o</sub> × [S<sub>t-1</sub>,x<sub>t</sub>] + B<sub>o</sub>)

Output<sub>t</sub> = O<sub>t</sub> × tanh(Cell<sub>t</sub>)
</strong>

And that Output<sub>t</sub> is what is output into the Recurrent Network.

<p align="center">
	<img width="300px" src="Images/img3.png">
	The Logistic Function plotted
</p>

As mentioned many times, all three gates are logistic. The reason for this is because it is very easy to backpropagate through them, and as such, it is possible for the model to learn exactly _how_ it is supposed to use this structure. This is one of the reasons for which LSTM is a very strong structure. Additionally, this solves the gradient problems by being able to manipulate values through the gates themselves -- by passing the inputs and outputs through the gates, we have now a easily derivable function modifying our inputs.

In regards to the problem of storing many states over a long period of time, LSTM handles this perfectly by only keeping whatever information is necessary and forgetting it whenever it is not needed anymore. Therefore, LSTMs are a very elegant solution to both problems.

Head [Here](Notebook.ipynb) to see the code.

## Thanks for Reading :)
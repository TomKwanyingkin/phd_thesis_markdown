# Introduction

## Background:

### Deep learning models:

Basically, deep learning models are artificial neural networks which can imitate the workings of the human brain in processing data and creating patterns for use in decision making and recognition. It can be well presented in a directed acyclic graph as shown as fig.1. Since a large and deep neural network has many layers and nodes in each layers, there will be numerous parameters need to be updated and tuned. Therefore, deep learning is gaining more and more popularity due to the Big Data and the existence of more powerful computers.

<center>
<img src="https://user-images.githubusercontent.com/56287097/112454163-5fafd580-8d93-11eb-9cd2-23b825f0d382.png" /><br>
Fig.1 A three-layer Neural Networks (NN)</center>

### Convolutional Neural Networks
s
The Convolutional Neural Networks (CNNs) is one of the deep learning models mainly deal with the image classification which will be implemented in the project. It is a type of feed-forward neural networks as illustrate in fig.2. The working principle of it is automatically learning and extracting features from the raw image data inputs in the training datasets. In each epoch, the network will do the forward calculation to calculate the output. Then, the loss calculation will be done to obtain the difference between the ground-truth output which is label and the calculated output. Based on the difference, the backward propagation can be done to optimize the parameter. If the datasets are sufficient, the output will be very precise.


<center>
<img src="https://user-images.githubusercontent.com/56287097/112454391-98e84580-8d93-11eb-8744-7ea0e5ec5350.jpg" /><br>
Fig.2 Architecture of CNNs</center>


## Objectives:

Covid-19 has been associated with more than millions deaths worldwide due to its easy and multi-way spreading. It was found that Covid-19 can be attached to surfaces like elevator buttons and be transferred through contact. In light of this, the public would try to use key, or sharp object to press the elevator buttons. Therefore, it is always observed that the plastic coating is torn as shown in the fig.3. The buttons can be damaged easily. In short, it seems that the traditional procedure of taking a lift which is pressing the buttons is outdated and is necessary to be replaced.

<center>
<img src="https://raw.githubusercontent.com/TomKwanyingkin/hiuraunt/main/Picture3.png" /><br>
Fig.3 Broken Buttons </center>
<br>
In light of this, the project is designed to develop and establish a system applying the gesture recognition technology, so that it enables users can take the elevator to any floors without contacting with the buttons. What they need to do is just making the hand gestures that represents the floor number. After a few seconds, the system will recognize the hand gestures of a single user or even multiple users and take them to the final destination. This way, the chance of spreading virus would be greatly reduced due to its zero-contact design. Even though the pandemic is over, the conveniences and innovation of this application can be still implemented in the shopping mall since the brad-new interaction with an elevator is appealing to customers.
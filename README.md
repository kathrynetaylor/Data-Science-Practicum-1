# Data-Science-Practicum-1

Introduction:
Hello! My name is Kathryne Taylor and this is my MSDS692 Data Science Practicum I Project. My project was a Semiconductor Deep Learning Project, where I collaborated with my company to create several image classification neural networks. I developed three deep learning models that would allow my company to gain a better understanding of semiconductor wafer patterns and material slicing stress fragments. This project was a deep dive into deep learning, where I gained a better understanding of Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and General Adversarial Networks (GANs).

Purpose:
The semiconductor solutions process to develop high-class and state-of-the-art semiconductor solutions process depends on hundreds of several complex manufacturing processes. Each widget that is processed also undergoes extreme stress tests based on quality control guidelines. Each of these processes and tests lead to high yield loss and lower gross margins for the company. By rendering thousands of semiconductor images and classifying each based on fragment diagnosis, my company can potentially benefit from a deep learning model that can pre-diagnosis semiconductor solution images before sending the widgets to the expensive test phase of the process. By automating the detection of wafer fault patterns in production, my company can also increase yield times and improve gross margins by cutting costs. 

Code:
My company has deemed the images as proprietary, so the code in this repository is showing similar models on the MNIST dataset. This is a handwritten digit dataset containing 60,000 training images and is available in the Keras Library. 

Result:
In order to produce a deep learning classifier for faults in wafer images, I developed three neural networks that extract image features. The first model was a sequential Convolutional Neural Network (CNN). The CNN model used several neural network techniques such as MaxPooling, Dropout,  and BatchNormalization to help the training process and avoid overfitting. The model overall had ~80% accuracy at detecting faults in the images. The second model I created is a LSTM Recurrent Neural Network (RNN). The RNN model is a powerful sequential model takes an input and the previous hidden state output of a neural network at each time step T. The LSTM (or Long Short Term Memory) contain input gates optionally allow information through in order to speed up computational time. The model was not as successful at classifying images as the CNN. Now that I have a better understanding of RNN’s and LSTM models, I would like to propose a different RNN model that can extract features from time series data of wafer production process steps (rather than a basic image classification problem). The last model I developed was a generative adversarial network (GAN). The GAN model is composed to two neural networks than train in a double feedforward and backpropagation loop. The first network, the generator, creates sample images from a latent noise vector that will eventually look almost identical to the original dataset. The second neural network, the discriminator, attempts to differentiate between the original image dataset and synthetic images from the generator. The model I built for classifying semiconductor solution images currently has a generative loss of ~10 and a discriminative loss of ~2 after training on 100,000 epochs. I would like to keep developing the model and run on longer training times or increase the number of epochs to obtain better results at classifying images. 



References:
Deep Learning and CNN’s
Hands on Machine Learning with Scikit-Learn & TensorFlow by Aurélien Géron (O'Reilly). CopyRight 2017 Aurélien Géron
Machine Learning - Over-& Undersampling - Python/ Scikit/ Scikit-Imblearn by Coding-Maniac
auprc, 5-fold c-v, and resampling methods by Jeremy Lane (Kaggle Notebook)
Goodfellow, I., Bengio, Y., & Courville, A. (2017). Deep learning. Cambridge, Mass: The MIT Press.
Chollet, F. Deep learning with Python.
Basic classification: Classify images of clothing  |  TensorFlow Core. (2020). Retrieved 6 March 2020, from https://www.tensorflow.org/tutorials/keras/classification

GAN’s

Auxiliary Classifier GAN - Keras Documentation. (2020). Retrieved 6 March 2020, from https://keras.io/examples/mnist_acgan/
Chen, C., Mu, S., Xiao, W., Ye, Z., Wu, L., & Ju, Q. (2019). Improving Image Captioning with Conditional Generative Adversarial Nets. Proceedings Of The AAAI Conference On Artificial Intelligence, 33, 8142-8150. doi: 10.1609/aaai.v33i01.33018142
Ravinutala, S. (2020). Playing around with SGDR. Retrieved 6 March 2020, from https://sidravi1.github.io/blog/2018/04/25/playing-with-sgdr
Brownlee, J. (2020). How to Develop a GAN for Generating MNIST Handwritten Digits. Retrieved 6 March 2020, from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
Data Augmentation tasks using Keras for image data. (2020). Retrieved 6 March 2020, from https://medium.com/@ayman.shams07/data-augmentation-tasks-using-keras-for-image-data-and-how-to-use-it-in-deep-learning-d4dd24e8ca19

RNN’s / LSTM’s:

CS 230 - Recurrent Neural Networks Cheatsheet. (2020). Retrieved 6 March 2020, from https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
Recurrent Neural Network (LSTM) · TensorFlow Examples (aymericdamien). (2020). Retrieved 6 March 2020, from https://wizardforcel.gitbooks.io/tensorflow-examples-aymericdamien/3.06_recurrent_network.html
Understanding LSTM Networks -- colah's blog. (2020). Retrieved 6 March 2020, from https://colah.github.io/posts/2015-08-Understanding-LSTMs/
What is a Recurrent NNs and Gated Recurrent Unit (GRUS). (2020). Retrieved 6 March 2020, from https://medium.com/@george.drakos62/what-is-a-recurrent-nns-and-gated-recurrent-unit-grus-ea71d2a05a69
Roger Grosse. Lecture 15: Exploding and Vanishing Gradients. (2020). Retrieved 6 March 2020, from http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf


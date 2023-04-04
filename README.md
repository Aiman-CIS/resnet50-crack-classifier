# resnet50-crack-classifier


This is a MATLAB code that builds a crack detection classifier using transfer learning with a pre-trained ResNet50 network.
The code first loads the dataset using imageDatastore and then splits it into training, validation, and test sets using splitEachLabel. It then loads the pre-trained ResNet50 network and extracts the layer names for feature extraction and classification.
Next, the code sets the input image size and creates an imageInputLayer to inherit the data augmentation and normalization settings from the pre-trained network. It also extracts the weight parameters of the second layer, which is the convolutional layer in the pre-trained network, and then creates a new convolutional layer to replace it.
A fully connected layer is added to the end of the feature extractor and a classification layer is added to perform the final classification. The layerGraph function is used to create a graph object representing the layers of the network, and the replaceLayer function is used to replace the original layers with the new layers.
After creating the network, the code sets up data augmentation and normalization for the training, validation, and test sets using augmentedImageDatastore. It then sets the training options using trainingOptions, which specifies the optimizer, mini-batch size, maximum number of epochs, initial learning rate, validation data, and other options for training the network. Finally, the trainNetwork function is called to train the network.
Once the network is trained, the code uses classify to predict labels and calculate accuracy, and plotconfusion to plot the confusion matrix. Finally, it displays four randomly selected images with their predicted labels.



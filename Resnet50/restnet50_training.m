
imds = imageDatastore('C:\Users\Hp\Desktop\Dataset\proc Dataset', ...
    'IncludeSubfolders',true, 'LabelSource','foldernames'); % this for labeling by folder names-

[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.15,0.15);%splitEachLabel function is designed to split an imageDatastore
net = resnet50(); 
numClasses = numel(categories(imdsTrain.Labels)); %number of unique classes 

% extract the names of the layers --main components of the feature extraction 
Input_Image=net.Layers(1).Name;
New_Conv_W=net.Layers(2).Name;
Feature_Learner=net.Layers(175).Name;
Output_Classifier=net.Layers(177).Name;


imageSize = [224 224 ]; % you can use here the original dataset size
global GinputSize
GinputSize = imageSize;


%inherits the data augmentation and normalization settings from the original network.
%one color channel--(gray-scale image)--[imageSize 1]
inputLayer = imageInputLayer([imageSize 1], 'Name', net.Layers(1).Name,... 
    'DataAugmentation', net.Layers(1).DataAugmentation, ... 
    'Normalization', net.Layers(1).Normalization);

newConv1_Weights = net.Layers(2).Weights; %extracts the weight parameters of the second layer.
newConv1_Weights = mean(newConv1_Weights(:,:,1:3,:), 3); % taking the mean of kernal channels--
%the weights of the first convolutional layer of the pre-trained network are averaged 
%across the RGB channels (the third dimension) to obtain a single gray-scale channel.
%This is necessary because the pre-trained network is trained on RGB images,
%but the new dataset may be gray-scale or may have fewer than 3 channels. 
%By averaging the weights across channels, we obtain a single set of weights that 
%can be applied to any input image regardless of its number of channels.


%new convolutional layer
newConv1 = convolution2dLayer(net.Layers(2).FilterSize(1), net.Layers(2).NumFilters,...
    'Name', net.Layers(2).Name,...
    'NumChannels', inputLayer.InputSize(3),... %inputLayer.InputSize returns a vector of format [height, width, channels], 
    'Stride', net.Layers(2).Stride,... %Stride' value of the original layer is accessed
    'DilationFactor', net.Layers(2).DilationFactor,...
    'Padding', net.Layers(2).PaddingSize,...
    'Weights', newConv1_Weights,...
    'Bias', net.Layers(2).Bias,...
    'BiasLearnRateFactor', net.Layers(2).BiasLearnRateFactor);


New_Feature_Learner=fullyConnectedLayer(num_Of_Classes,...
    'Name','Crack Feature Learner',...
   'WeightLearnRateFactor' , 10);

 New_Classification_Layer=classificationLayer('Name','Crack Classifier');
 
 
 Network_Architecture=layerGraph(net);
 
 New_Network=replaceLayer( Network_Architecture,Input_Image,inputLayer);
 New_Network=replaceLayer( New_Network,New_Conv_W,newConv1);
 New_Network=replaceLayer( New_Network,Feature_Learner,New_Feature_Learner);
 New_Network=replaceLayer(New_Network,Output_Classifier,New_Classification_Layer);
 
 
analyzeNetwork(New_Network)

%  First argument specifies output size of the augmented image
%  An augmented image datastore applies random transformations to the input images, 
%  such as rotations, translations, and scaling, to increase the number of
%  training examples .


augimdsTrain = augmentedImageDatastore([imageSize 1],imdsTrain);
augimdsValidation = augmentedImageDatastore([imageSize 1],imdsValidation);
augimdsTest = augmentedImageDatastore([imageSize 1],imdsTest);


 options = trainingOptions('adam', ...
    'MiniBatchSize',10,... %number of training examples in each batch of the data. Here, each batch contains 10 images.
    'MaxEpochs',5, ...
    'InitialLearnRate',0.001, ...
    'Shuffle','every-epoch', ... %Specifies how the training data is shuffled at the beginning of each epoch. 
    'ValidationData',augimdsValidation, ...% The validation data to use during training
    'ValidationFrequency',3, ...%The frequency at which to evaluate the network performance on the validation set. Here, it is set to evaluate the network after every 3 epochs.
    'Verbose',false, ...%Specifies whether or not to display training progress information in the command window.
    'Plots','training-progress');

net1=trainNetwork( augimdsTrain,New_Network,options);

save net1 %saves the trained network (net1) 
 


%if the training set has 100 images, and the batch size is set to 10, 
%then there will be 10 iterations per epoch, and each iteration will process 10 images.
 
  
 

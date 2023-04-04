Testing=imageDatastore("C:\Users\Hp\Desktop\Dataset\cat");
imageSize = [224 224 ]; % you can use here the original dataset size
global GinputSize
GinputSize = imageSize;
resizeTest = augmentedImageDatastore([imageSize 1],Testing);

[Predicted_Label,Probability]=classify(net1,resizeTest);%The output of the "classify"---
%function is two arrays: "Predicted_Label" and "Probability". 
%takes a trained network and input data.


index=randperm(numel(Testing.Files),4);
%Testing.Files contains the file paths for the test images, and numel(Testing.Files) 
%returns the total number of test images.Using randperm(numel(Testing.Files), 4) 
%generates a random permutation of the integers from 1 to the number of elements in
%Testing.Files and returns the first 4 elements of this permutation. These 4 elements
%represent random indices into the Testing.Files cell array, and correspond to the file paths
%for a random subset of 4 test images.These randomly selected test images can then be loaded
%into memory and used to evaluate the performance of a machine learning model.

figure
for i = 1:4
    subplot(2,2,i)
    I=readimage(Testing,index(i));
    imshow(I)
    label=Predicted_Label(index(i));
    title(string(label)+","+num2str(100*max(Probability(index(i),:)),3)+"%");
end
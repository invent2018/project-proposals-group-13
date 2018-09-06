%for rerunning the training OR during actual presentation


%CHECK LINES 5 AND 26 AND 35
addpath(genpath('C:\Users\Victor Loh\Desktop\boosted')) %add path of where the whole deepfashion data and imdsimport file
load('data.mat') %whatever the file name is called

net = vgg16;


LayersTransfer = net.Layers(1:end-3);
Layers = [
    LayersTransfer
    fullyConnectedLayer(24,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


%------------------------------randomising ind-------------------------
[trainInd,testInd] = dividerand(height(actualdataset),0.9,0.1); %First ratio is no. of training, second is test
trainingData = actualdataset(trainInd,:);
testData = actualdataset(testInd:end,:);


options = trainingOptions('sgdm', ...
  'LearnRateSchedule','piecewise',...
  'LearnRateDropFactor',0.1,...
  'LearnRateDropPeriod',5,...
  'MiniBatchSize', 128, ...
  'InitialLearnRate', 1e-4, ...
  'MaxEpochs', 20);



%options still not optimised, have to see how things go

[detector1,traininginfo] = trainRCNNObjectDetector(trainingData,Layers, options); %Change the name also for network also
%this is the command to actually train so make this a comment if you dont
%want to actually train yet^^


trainedVGG16 = detector1; 
clearvars clothingdataset i labelarray labellist labelsize labeltable labelunique rootFolder startRow testData training testInd trainInd ans A

save ('VGG16_1.mat') %CHANGE THIS ALSO





img = imread('test1.jpg');
[bbox, score, label] = detect(detector1, img);

%--------------- thresholding
sizearray = size(score);
threshold = score > 0.62;
n = 1;
for i = 1 : sizearray(1)
    if threshold(i) == true
    filterscore(n,:) = score(i,:);
    filterbbox(n,:) = bbox(i,:);
    filterlabel(n,:) = label(i,:);
    n = n+1;
    end
    
end
annotation = cell(n-1,1);
for i = 1 : n-1
    annotation{i} = sprintf('%s: (Confidence = %f)', filterlabel(i), filterscore(i));
end


detectedImg = insertObjectAnnotation(img, 'rectangle', filterbbox, annotation);

figure(2)
imshow(detectedImg)





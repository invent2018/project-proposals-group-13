%for rerunning the training OR during actual presentation


%CHECK LINES 5 AND 26 AND 35
addpath(genpath('C:\Users\Victor Loh\Desktop\boosted')) %add path of where the whole deepfashion data and imdsimport file
load('trainedDetector3.mat') %whatever the file name is called


Layers = trainedDetector3.Network.Layers;
Layers(23).WeightLearnRateFactor = 1;
Layers(23).BiasLearnRateFactor = 1;

%------------------------------randomising ind-------------------------
[trainInd,testInd] = dividerand(height(actualdataset),0.3,0.7 ); %First ratio is no. of training, second is test
trainingData = actualdataset(trainInd,:);
testData = actualdataset(testInd:end,:);


options = trainingOptions('sgdm', ...
  'LearnRateSchedule','piecewise',...
  'LearnRateDropFactor',0.1,...
  'LearnRateDropPeriod',1,...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 4);



%options still not optimised, have to see how things go

[detector1,traininginfo] = trainRCNNObjectDetector(trainingData,Layers, options); %Change the name also for network also
%this is the command to actually train so make this a comment if you dont
%want to actually train yet^^


trainedDetector3 = detector1; 

save ('trainedDetectorFINAL_1.mat') %CHANGE THIS ALSO





img = imread('test1.jpg');
[bbox, score, label] = detect(detector1, img);
detectedImg = insertShape(img, 'Rectangle', bbox);
figure(1)
imshow(detectedImg)

[score, idx] = max(score);
score
label

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure(2)
imshow(detectedImg)





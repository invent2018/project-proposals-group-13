%for rerunning the training OR during actual presentation


%CHECK LINES 5 AND 26 AND 35
addpath(genpath('C:\Users\Victor Loh\Desktop\boosted')) %add path of where the whole deepfashion data and imdsimport file
load('trainedDetector3.mat') %whatever the file name is called


Layers = trainedDetector3.Network.Layers;
Layers(23).WeightLearnRateFactor = 1;
Layers(23).BiasLearnRateFactor = 1;

%------------------------------randomising ind-------------------------
[trainInd,testInd] = dividerand(height(actualdataset),0.25,0.75); %First ratio is no. of training, second is test
trainingData = actualdataset(trainInd,:);
testData = actualdataset(testInd:end,:);


options = trainingOptions('sgdm', ...
  'LearnRateSchedule','piecewise',...
  'LearnRateDropFactor',0.1,...
  'LearnRateDropPeriod',1,...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 8);



%options still not optimised, have to see how things go

[detector1,traininginfo] = trainRCNNObjectDetector(trainingData,Layers, options); %Change the name also for network also
%this is the command to actually train so make this a comment if you dont
%want to actually train yet^^


trainedDetector_final = detector1; 

save ('trainedDetectorFINAL_1.mat') %CHANGE THIS ALSO





%--------------------testing----------------------------
img = imread('test1.jpg');
[bbox, score, label] = detect(detector1, img);

%--------------- thresholding-------------------------------
sizearray = size(score);
threshold = score > 0.8;   %Arbitrarily set
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


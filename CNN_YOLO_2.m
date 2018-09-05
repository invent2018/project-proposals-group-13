
%2/9/18 - ROIcolor region proposal -> CNN identify object 


clear all
clc

rootFolder = 'C:\Users\kai10_000\Desktop\SSD leggo\DATA\Deepfashion\In-shop Clothes Retrieval Benchmark\Img';



%------------------------------Deepfashion importing---------------------------
global actualdataset;
imdsimport
disp('imdsdone') 


%----------image file name
temp = size(clothingdataset);
clothingdataset = clothingdataset(3:temp(1),:);
actualdataset.imagefilename = fullfile(rootFolder, clothingdataset.imagefilename);
clearvars temp
disp('imagefilenamedone') 



%------------table
actualdataset = struct2table(actualdataset);



%-------------labels
expression1 = 'img/';
expression2 = '/id';
expression3 = '/';
for i = 1 : height(clothingdataset)
    
    str = clothingdataset.imagefilename{i};

    splitstr = regexp(regexp(str,expression1,'split'),expression2,'split');
    splitstr = regexp(splitstr{2}{1},expression3,'split');
    label = join(splitstr, '_');
    label = label{1};
    labellist{i,1} = label;
    
    
end
labelunique = unique(labellist);
labelsize = size(labelunique);

clearvars  label expression1 expression2 expression3 str splitstr i 
disp('labels done')




labelarray = zeros(height(clothingdataset),labelsize(1)); %can be nan also see how
labelarray = num2cell(labelarray);
%----------bboximport
   for i = 1 :height(clothingdataset)
    A = zeros(1,4);
    A(1,1) = clothingdataset.x1(i); %top left
    A(1,2) = clothingdataset.y1(i); %top left
    A(1,3) = clothingdataset.x2(i) - clothingdataset.x1(i); %width
    A(1,4) = clothingdataset.y2(i) - clothingdataset.y1(i); %height
    x = strmatch(labellist{i}, cellstr(labelunique));
   labelarray(i,x) = {A};
    
    %actualdataset.bbox{i} = A;
    
   end
 labeltable = array2table(labelarray,'VariableNames', labelunique'); 
 
 actualdataset = [actualdataset labeltable];
 %actualdataset.bbox = actualdataset.bbox'; %conjugate transposing of table


disp('bboxdone')  

disp('datadone')
%-------------------------data preprocessing for CNN-------------------------

idx1 = floor(0.5 * height(actualdataset));
idx2 = floor(0.8* height(actualdataset));
trainingData = actualdataset(1:100,:);
validationData = actualdataset(idx1:idx2,:);
testData = actualdataset(idx2:end,:);






%----------------------CNN architecture (faster rcnn) -----------------------

net = load('yolonet.mat');
out = net.yolonet; 
layersTransfer = out.Layers(1:end-3);
numClasses = labelsize(1);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses+1,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%options = trainingOptions('sgdm', ...
   % 'Momentum', 0.9, ...
    %'MiniBatchSize',128, ...
  %  'MaxEpochs',20, ...
   % 'InitialLearnRate',1e-4, ...
    %'ValidationData',validationData, ...
    %'ValidationFrequency',3, ...
    %'ValidationPatience',Inf, ...
    %'Verbose',false, ...
    %'Plots','training-progress');

% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%[trainedNet,traininfo] = trainNetwork(trainingData,layers,options)
 



%detector = trainFasterRCNNObjectDetector(trainingData, out, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'BoxPyramidScale', 1.2);


%3/9/18 - RUN 2

clear all
clc
addpath(genpath('C:\Users\Victor Loh\Desktop\boosted')) %add path of where the whole deepfashion data and imdsimport file

rootFolder = 'C:\Users\Victor Loh\Desktop\boosted';
%change rootfolder to dir of your In-shop Clothes Retrieval Benchmark\Img file



%------------------------------Deepfashion importing---------------------------
global actualdataset;
imdsimport 
disp('imdsdone') 


%----------image file name
temp = size(clothingdataset);
clothingdataset = clothingdataset(3:temp(1),:);
actualdataset.imagefilename = convertStringsToChars(fullfile(rootFolder, clothingdataset.imagefilename));
clearvars temp
disp('imagefilenamedone') 



%------------table
actualdataset = struct2table(actualdataset);



%-------------labels
expression1 = 'img/';
expression2 = '/id';
expression3 = '/';
labellist = num2cell(zeros(height(clothingdataset),1));
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




labelarray = nan(height(clothingdataset),labelsize(1)); %can be nan also see how
labelarray = num2cell(labelarray);
labelarray(cellfun(@isnan,labelarray)) = {[]};
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

[trainInd,testInd] = dividerand(height(actualdataset),0.2,0.8); %First ratio is no. of training, second is test
trainingData = actualdataset(trainInd,:);
testData = actualdataset(testInd:end,:);



%----------------------CNN architecture (rcnn) -----------------------
net1 = load('trainedDetector.mat');

options = trainingOptions('sgdm', ...
  'LearnRateSchedule','piecewise',...
  'LearnRateDropFactor',0.2,...
  'LearnRateDropPeriod',2,...
  'MiniBatchSize', 16, ...
  'InitialLearnRate', 1e-4, ...
  'MaxEpochs', 10);



%options still not optimised, have to see how things go

[detector1,traininginfo] = trainRCNNObjectDetector(trainingData, net1, options);
%this is the command to actually train so make this a comment if you dont
%want to actually train yet^^

trainedDetector2 = detector1; 
clearvars clothingdataset i labelarray labellist labelsize labeltable labelunique rootFolder startRow testData training testInd trainInd ans A
save ('trainedDetector2.mat')

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
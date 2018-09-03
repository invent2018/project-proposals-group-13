
%2/9/18 - ROIcolor region proposal -> CNN identify object 


clear all
clc
addpath(genpath('C:\Users\kai10_000\Desktop\SSD leggo'))
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

idx1 = floor(0.7 * height(actualdataset));
trainingData = actualdataset(1:idx1,:);
testData = actualdataset(idx1:end,:);




%----------------------CNN architecture (rcnn) -----------------------


options = trainingOptions('sgdm', ...
  'MiniBatchSize', 128, ...
  'InitialLearnRate', 1e-4, ...
  'MaxEpochs', 24);

%[trainedNet,traininfo] = trainNetwork(trainingData,layers,options)

[detector1,traininginfo] = trainRCNNObjectDetector(trainingData, vgg16, options);


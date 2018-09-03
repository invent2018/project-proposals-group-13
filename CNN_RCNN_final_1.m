
%3/9/18 - CNN - RCNN object detection hopefully this fucking works

% 1. download
% https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc 
% the whole thing, except for img_highres.zip shit in the "img" folder
% 2. download imdsimport.m (script file for matlab) but dunnid open just
% make sure its added to path (i.e. added to the entire dataset folder somewhere

% 3. just read through the comments of entire script here, and change what
% is necessary 

% 4. Press run on top but dun if u dont want train u can make last command
% a comment first

% 5. after you are done, type in this command:
    %save detector1
 
% 6. you should be getting a detector1.mat file in the directory on the
%left, just somehow send it to me thanks
 
 
clear all
clc
addpath(genpath('C:\Users\kai10_000\Desktop\SSD leggo\DATA\Deepfashion')) %add path of where the whole deepfashion data and imdsimport file

rootFolder = 'C:\Users\kai10_000\Desktop\SSD leggo\DATA\Deepfashion\In-shop Clothes Retrieval Benchmark\Img';
%change rootfolder to dir of your In-shop Clothes Retrieval Benchmark\Img file



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

idx1 = floor(0.5 * height(actualdataset));
idx2 = floor(0.8* height(actualdataset));
trainingData = actualdataset(1:idx1,:);
validationData = actualdataset(idx1:idx2,:);
testData = actualdataset(idx2:end,:);




%----------------------CNN architecture (rcnn) -----------------------


options = trainingOptions('sgdm', ...
  'MiniBatchSize', 128, ...
  'InitialLearnRate', 1e-4, ...
  'MaxEpochs', 24);
%options still not optimised, have to see how things go

[detector1,traininginfo] = trainRCNNObjectDetector(trainingData, vgg16, options); 
%this is the command to actually train so make this a comment if you dont
%want to actually train yet^^


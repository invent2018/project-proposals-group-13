
%2-0/9/18 - ClusterLabs run 1





clear all
clc
addpath(genpath('C:\Users\temp\Desktop\boosted')) %add path of where the whole deepfashion data and imdsimport file

rootFolder = char('C:\Users\temp\Desktop\boosted\resized');
%change rootfolder to dir of your In-shop Clothes Retrieval Benchmark\Img file



%------------------------------Deepfashion importing---------------------------
global actualdataset;
imdsimport 
disp('imdsdone') 


%----------image file name
temp = size(clothingdataset);
clothingdataset = clothingdataset(3:temp(1),:);
actualdataset = struct;
    for i = 1 : height(clothingdataset)
        x = char(clothingdataset.imagefilename(i));
        actualdataset.imagefilename{i,1} = fullfile(rootFolder, char(clothingdataset.imagefilename{i}));
    end
    actualdataset = actualdataset';

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
    A(1,1) = clothingdataset.x1(i) *(224/256); %top left
    A(1,2) = clothingdataset.y1(i) *(224/256); %top left
    A(1,3) = (clothingdataset.x2(i) - clothingdataset.x1(i))*(224/256); %width
    A(1,4) = (clothingdataset.y2(i) - clothingdataset.y1(i))*(224/256); %height
    x = strmatch(labellist{i}, cellstr(labelunique));
   labelarray(i,x) = {A};
   bboxarray(i,1) = {A};
    %actualdataset.bbox{i} = A;
    
   end
 labeltable = array2table(labelarray,'VariableNames', labelunique'); 
 
 actualdataset = [actualdataset labeltable];
 %actualdataset.bbox = actualdataset.bbox'; %conjugate transposing of table


disp('bboxdone')  

disp('datadone')


%--------------------feature %extraction-------------------------------


lbp = nan(height(clothingdataset),1);
colormoment = lbp;
hog = lbp;
for i = 1: height(clothingdataset)
    
    I = imread(actualdataset.imagefilename{i});
    p = bboxarray{i,1};
featurethanks_final_updated(I,p,i);

end















%--------------------------detection-------------------------------
trainedVGG16 = detector1; 
clearvars  clothingdataset i labelarray labellist labelsize labeltable labelunique rootFolder startRow testData training testInd trainInd ans A



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

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



load trainedDetector3
clearvars trainedDetector3 trainedDetector2 trainedDetector detector1
rootFolder = 'C:\Users\Victor Loh\Desktop\boosted\resized';
mkdir(fullfile(rootFolder,'img', 'MEN'))
mkdir(fullfile(rootFolder,'img', 'WOMEN'))
addpath(genpath('C:\Users\Victor Loh\Desktop\boosted'))

%change rootfolder to dir of your In-shop Clothes Retrieval Benchmark\Img file


temp = size(clothingdataset);
clothingdataset = clothingdataset(3:temp(1),:);
actualdataset.imagefilename = convertStringsToChars(fullfile(rootFolder, clothingdataset.imagefilename));
clearvars temp
disp('imagefilenamedone') 

%------------resizing images----------------------

for i = 1 : height(actualdataset)
    
   I = imread(actualdataset{i,1}{1});
   I = imresize(I, [224 224]);
   filename = fullfile(rootFolder, clothingdataset.imagefilename{i});
   
   actualdataset.imagefilename{i} = convertStringsToChars(fullfile(rootFolder, clothingdataset.imagefilename{i}));
   
   imwrite(I,filename, 'jpg') 
    
    
end    

disp('resizing done')

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
    
    %actualdataset.bbox{i} = A;
    
   end
 labeltable = array2table(labelarray,'VariableNames', labelunique'); 
 
 actualdataset = [actualdataset labeltable];
 %actualdataset.bbox = actualdataset.bbox'; %conjugate transposing of table


disp('bboxdone')  

disp('datadone')


actualdataset = struct2table(actualdataset); %------------ table


save data
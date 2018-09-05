
%2/9/18 - ROIcolor region proposal -> CNN identify object -> attribute
%prediction from HOG or SIFT


clear all
clc

rootFolder = 'C:\Users\kai10_000\Desktop\SSD leggo\DATA\Deepfashion\In-shop Clothes Retrieval Benchmark\Img';



%------------------------------Deepfashion importing---------------------------
global actualdataset;
imdsimport


disp('imdsdone') 

%image file name
temp = size(clothingdataset);
clothingdataset = clothingdataset(3:temp(1),:);
actualdataset.imagefilename = fullfile(rootFolder, clothingdataset.imagefilename);


disp('imagefilenamedone') 

%bboximport
   for i = 1:temp(1)-2
    A = zeros(1,4);
    A(1,1) = clothingdataset.x1(i); %top left
    A(1,2) = clothingdataset.y1(i); %top left
    A(1,3) = clothingdataset.x2(i) - clothingdataset.x1(i); %width
    A(1,4) = clothingdataset.y2(i) - clothingdataset.y1(i); %height
    actualdataset.bbox{i} = A;
    
   end
 actualdataset.bbox = actualdataset.bbox'; %conjugate transposing of table

disp('bboxdone')  
 
 %color import
 importcolor
 actualdataset.color = colorlist(:,1);
 
disp('colordone') 


%attrib/labels import
attriblistimport;
attribimport1 %until 4000
disp('attrib1 done')
attribimport2
disp('attrib2 done')
attr = outerjoin(attr1,attr2,'MergeKeys',true);
attr = unique(attr);

attribcell = zeros(size(attr)); 
attribcell = num2cell(attribcell);%cell array of labels
for i = 1:height(attr)
    x = 1;
    for n = 1:height(attriblist)
        if attr{i,n+1} == 1
            attribcell{i,x} = attriblist{n,1};
            x = x + 1;
        end
    end
end
disp('attribcell done')


%creating an idlist
evallistimport
B = sortrows(evallist, 2);
actualdataset.id = B(:,2);
idlist = B.VarName2;
clearvars B evallist

disp('idlist done')

x = 1;
%transfering attribcell into actualdataset
for i = 1:height(idlist) %5272??
    for p = 1:height(attribcell) %7928
        if idlist(i) == p
            for n = 1:height(attriblist) %464
                if isempty(attribcell{p,n}) == false
                    temp{x,1} = attribcell{p,x};
                    x = x + 1;
                else
                    x = 1;
                end
            end 
        end
    end
end
numClasses = height(attriblist);
clearvars attr attriblist x
 disp('attricell transfer done')
        
    


%table
actualdataset = struct2table(actualdataset);

disp('datadone')
clearvars A clothingdataset colorlist temp

%-------------------------CNN-----------------------------------------------

nnet = load('yolonet.mat');
out = nnet.yolonet; 
layersTransfer = out.Layers(1:end-3);
numClasses = 2000;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];





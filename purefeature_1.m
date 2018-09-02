
%1/9/18 - ROIcolor region proposal -> CNN identify object -> attribute
%prediction from HOG or SIFT


clear all
clc

rootFolder = 'C:\Users\kai10_000\Desktop\SSD leggo\DATA\Deepfashion\In-shop Clothes Retrieval Benchmark\Img';



net = vgg16;
layerstransfer = net.Layers(1:end-4);


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


attribcell = zeros(8009,41); 
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

x = 1;
%transfer into actualdataset
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
clearvars attr attriblist x
 disp('attricell transfer done')
        
    
evallistimport
B = sortrows(evallist, 2);
actualdataset.id = B(:,2);
idlist = B.VarName2;
clearvars B evallist

%table
actualdataset = struct2table(actualdataset);

disp('datadone')
clearvars A clothingdataset colorlist temp




%---------------------------feature extraction loop-----------------------------
colormoment = cell(height(actualdataset), 1);
lbp = cell(height(actualdataset), 1);
hog = cell(height(actualdataset), 1);


for p = 3009 %height(actualdataset)
    
    I = imread(char(actualdataset.imagefilename(p)));
    I = imcrop(I,actualdataset.bbox{p});
figure(p)
imshow(I)

    
Iroi = I;

numcolors = 30000;   %arbitrary assigned
[X,map] = rgb2ind(I, numcolors);
BW = roicolor(X,100,1000);
%imshow(BW)


A = size(I);
for i = 1: A(1)
    
    for n = 1:A(2)
        
        
        if BW(i,n) == 0
             Iroi(i,n,1) = 0;
             Iroi(i,n,2) = 0;
             Iroi(i,n,3) = 0;
    
         end
    end
end

%figure(p) 
%imshow(Iroi)
%------------------------hog------------------------------
[hogfeature,hogVisualization] = extractHOGFeatures(Iroi);

figure(p);
imshow(Iroi); 
hold on;
plot(hogVisualization);


%---------------------lbp--------------------------------

Igray = rgb2gray(Iroi);
lbpfeature = extractLBPFeatures(Igray);

%numNeighbors = 8;
%numBins = numNeighbors*(numNeighbors-1)+3;
%featureview = reshape(features,numBins,[])

%featureview = reshape(features, 



%----------------------colormoment---------------------
% Extract RGB Channel
R=Iroi(:,:,1);
G=Iroi(:,:,2);
B=Iroi(:,:,3);
% Extract Statistical features
% 1] MEAN
meanR=mean2(R);
meanG=mean2(G);
meanB=mean2(B);
% 2] Standard Deviation
stdR=std2(R);
stdG=std2(G);
stdB=std2(B);
colorfeature = [meanR meanG meanB stdR stdG stdB];

colormoment{p, 1} = colorfeature;
lbp {p, 1} = lbpfeature;
hog {p, 1} = hogfeature;


end

%clearvars 







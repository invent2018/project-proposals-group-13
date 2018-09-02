function featurethanks_final(I,p) %I is imread(fullfile());
[rows, columns, numberOfColorBands] = size(I);
%I is imreaded image

I = imcrop(I,actualdataset.bbox{p});
imshow(I)


    
Iroi = I;

numcolors = 30000;   %arbitrary assigned
[X,map] = rgb2ind(I, numcolors);
BW = roicolor(X,200,1000);
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
figure(p)    
imshow(Iroi)
%------------------------hog------------------------------
[hogfeature,hogVisualization] = extractHOGFeatures(Iroi);


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
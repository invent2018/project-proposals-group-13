function featurethanks_final(I,p,n) %I is imread(fullfile());

[rows, columns, numberOfColorBands] = size(I);
%I is imreaded image

xcoord = [p(:,1) p(:,1)+p(:,3) p(:,1)+p(:,3) p(:,1) p(:,1)];
ycoord = [p(:,2) p(:,2) p(:,2)+p(:,4) p(:,2)+p(:,4) p(:,2)];
Xdiff = p(:,3) - p(:,1);
Ydiff = p(:,2) - p(:,4);

L = superpixels(I,500); %arbitrarily set

ROImask = poly2mask(xcoord, ycoord, rows, colums);

     
BW = grabcut(I,L,ROImask)

maskedImage = I;
maskedImage(repmat(~BW,[1 1 3])) = 0;
figure;
imshow(maskedImage)


Iroi = imcrop(I,p);
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

colormoment{n, 1} = colorfeature;
lbp {n, 1} = lbpfeature;
hog {n, 1} = hogfeature;



end
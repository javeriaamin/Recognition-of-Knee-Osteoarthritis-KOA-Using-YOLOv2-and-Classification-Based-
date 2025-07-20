# Recognition-of-Knee-Osteoarthritis-KOA-Using-YOLOv2-and-Classification-Based-on Convolutional Neural Network
Usman Yunus  , Javeria Amin 2	, Muhammad Sharif 1, Mussarat Yasmin 1, Seifedine Kadry 3  and Sujatha Krishnamoorthy 4,5,*

# The 2D grayscale radiograph images are converted to 3-channel and LBP features are extracted having the dimension of N × 59 out of which the best features of N × 55 are selected using PCA. The deep features are also extracted using Alex-Net and Dark-net-53 with the dimensions of N × 1024 and N × 4096, respectively, where N represents the number of images. Then, N × 1000 features are selected individually from both models using PCA. Finally, the extracted features are fused serially with the dimension of N × 2055 and passed to the classifiers on a 10-fold cross-validation.

# Implementation Code for Classification

clc; clear; close all;

% Step 1: Load Images
imageFolder = ''D:\usman data set\knee data_After_Loop','recursive''; 
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

labels = imds.Labels;
inputSize = [227 227 3];
augImds = augmentedImageDatastore(inputSize, imds);

% Step 2: Load Pretrained Networks
alex = alexnet;
dark = darknet53;

% Step 3: Initialize Feature Matrices
alexFeatures = [];
darkFeatures = [];
lbpFeatures = [];

% Step 4: Feature Extraction
for i = 1:numel(imds.Files)
    img = readimage(imds, i);
    img_resized = imresize(img, inputSize);

    % ---- AlexNet: FC7 Layer (4096 features) ----
    af = activations(alex, img_resized, 'fc7', 'OutputAs', 'rows');
    alexFeatures = [alexFeatures; af];

    % ---- DarkNet-53: Global Avg Pool Layer (1024 features) ----
    df = activations(dark, img_resized, 'global_avg_pool', 'OutputAs', 'rows');
    darkFeatures = [darkFeatures; df];

    % ---- LBP (59 features) ----
    gray = rgb2gray(img_resized);
    lbp = extractLBPFeatures(gray, 'Upright', false); % returns 59-bin histogram
    lbpFeatures = [lbpFeatures; lbp];
end

% Step 5: PCA - Dimensionality Reduction
% AlexNet 4096 -> 1000
[~, alexScore] = pca(alexFeatures);
alexReduced = alexScore(:, 1:1000);

% DarkNet 1024 -> 1000
[~, darkScore] = pca(darkFeatures);
darkReduced = darkScore(:, 1:1000);

% LBP 59 -> 55
[~, lbpScore] = pca(lbpFeatures);
lbpReduced = lbpScore(:, 1:55);

% Step 6: Feature Fusion
fusedFeatures = [alexReduced, darkReduced, lbpReduced];  % N × 2055

% Step 7: 10-Fold Cross Validation
cv = cvpartition(labels, 'KFold', 10);
acc = zeros(cv.NumTestSets,1);

for k = 1:cv.NumTestSets
    trainIdx = training(cv, k);
    testIdx = test(cv, k);

    XTrain = fusedFeatures(trainIdx, :);
    XTest = fusedFeatures(testIdx, :);
    YTrain = labels(trainIdx);
    YTest = labels(testIdx);

    % Train SVM
    model = fitcecoc(XTrain, YTrain);

    % Predict
    predictions = predict(model, XTest);
    acc(k) = mean(predictions == YTest);

    fprintf("Fold %d Accuracy: %.2f%%\n", k, acc(k)*100);
end

% Final Accuracy
fprintf("\nAverage 10-Fold Accuracy: %.2f%%\n", mean(acc)*100);

% Optional: Confusion Matrix for Last Fold
confusionchart(YTest, predictions);
# Localization
inputsize=[224 224]
I = imread('grade4 (9).png');
I = imresize(I,inputsize(1:2));

[bboxes,scores] = detect(detectorTinyYolo2,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

[bboxes,scores,labels] = detect(detectorTinyYolo2,I)
if ~isempty(bboxes)    BBarea = bboxes(:,1).*bboxes(:,2)
    BBarea = bboxes(:,1).*bboxes(:,2)

    
    detectedI = insertObjectAnnotation(I,'rectangle',bboxes,cellstr(labels));
end
figure
imshow(detectedI)

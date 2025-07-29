clc; clear; close all;
rng(0);  % For reproducibility

% Step 1: Load Images
imageFolder = 'D:\usman data set\knee data_After_Loop'; 
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

labels = imds.Labels;

% Step 2: Load Pretrained Networks
alex = alexnet;
dark = darknet53;

% Step 3: Pre-extract all raw features (no PCA here)
alexFeatures = [];
darkFeatures = [];
lbpFeatures = [];

for i = 1:numel(imds.Files)
    img = readimage(imds, i);
    
    % Resize for AlexNet
    img_alex = imresize(img, [227 227]);
    af = activations(alex, img_alex, 'fc7', 'OutputAs', 'rows');
    alexFeatures = [alexFeatures; af];

    % Resize for DarkNet
    img_dark = imresize(img, [256 256]);
    df = activations(dark, img_dark, 'global_avg_pool', 'OutputAs', 'rows');
    darkFeatures = [darkFeatures; df];

    % LBP Features
    gray = rgb2gray(imresize(img, [227 227]));  % same as AlexNet size
    lbp = extractLBPFeatures(gray, 'Upright', false);
    lbpFeatures = [lbpFeatures; lbp];
end

% Step 4: 10-Fold Cross Validation with PCA inside each fold
cv = cvpartition(labels, 'KFold', 10);
acc = zeros(cv.NumTestSets,1);

for k = 1:cv.NumTestSets
    trainIdx = training(cv, k);
    testIdx = test(cv, k);

    % Split features and labels
    alexTrain = alexFeatures(trainIdx, :);
    alexTest  = alexFeatures(testIdx, :);

    darkTrain = darkFeatures(trainIdx, :);
    darkTest  = darkFeatures(testIdx, :);

    lbpTrain = lbpFeatures(trainIdx, :);
    lbpTest  = lbpFeatures(testIdx, :);

    YTrain = labels(trainIdx);
    YTest = labels(testIdx);

    % Step 5: PCA per fold (only trained on training set)
    [coeffA, scoreA, ~, ~, ~, muA] = pca(alexTrain);
    alexTrainReduced = scoreA(:,1:1000);
    alexTestReduced = (alexTest - muA) * coeffA(:,1:1000);

    [coeffD, scoreD, ~, ~, ~, muD] = pca(darkTrain);
    darkTrainReduced = scoreD(:,1:1000);
    darkTestReduced = (darkTest - muD) * coeffD(:,1:1000);

    [coeffL, scoreL, ~, ~, ~, muL] = pca(lbpTrain);
    lbpTrainReduced = scoreL(:,1:55);
    lbpTestReduced = (lbpTest - muL) * coeffL(:,1:55);

    % Step 6: Feature Fusion
    XTrain = [alexTrainReduced, darkTrainReduced, lbpTrainReduced];
    XTest = [alexTestReduced, darkTestReduced, lbpTestReduced];

    % Step 7: Train SVM
    t = templateSVM('KernelFunction', 'linear'); % or try 'rbf'
    model = fitcecoc(XTrain, YTrain, 'Learners', t);

    % Predict and Evaluate
    predictions = predict(model, XTest);
    acc(k) = mean(predictions == YTest);

    fprintf("Fold %d Accuracy: %.2f%%\n", k, acc(k)*100);
end

% Final Accuracy
fprintf("\nAverage 10-Fold Accuracy: %.2f%%\n", mean(acc)*100);

% Confusion Matrix for Last Fold
confusionchart(YTest, predictions);

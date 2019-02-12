tic
clc; clear all
data_matrix=xlsread('file1.xlsx');
icol = size(data_matrix,2)

data_predictor = data_matrix(:,1:icol-1); % predictors matrix
label = data_matrix(:,end); % last column is 2 for benign, 4 for malignant


rng default
c = cvpartition(length(data_predictor), 'HoldOut', 0.3);
% Extract the indices of the training and test sets.
trainIdx = training(c);
testIdx = test(c);
% Create the training and test data sets.
XTrain = data_predictor(trainIdx, :);
XTest = data_predictor(testIdx, :);
yTrain = label(trainIdx);
yTest = label(testIdx);
% Create an ensemble of 100 trees.
forestModel = fitensemble(XTrain, yTrain, 'Bag', 100,...
                            'Tree', 'Type', 'Classification'); 
% Predict and evaluate the ensemble model.
forestPred = predict(forestModel, XTest);
% errs = forestPred ~= yTest;
% testErrRateForest = 100*sum(errs)/numel(errs);
% display(testErrRateForest)
% Perform 10-fold cross validation.
cvModel = crossval(forestModel); % 10-fold is default 
cvErrorForest = 100*kfoldLoss(cvModel);
display(cvErrorForest)
% Confusion matrix.
C = confusionmat(yTest, forestPred);

BaggedEnsemble = generic_random_forests(forestPred, yTest, 500, 'classification')
X = sprintf('Test accuracy: %f\n', mean(forestPred==yTest));
disp(X)


figure(figOpts{:})
imagesc(C)
colorbar
colormap('cool')
[Xgrid, Ygrid] = meshgrid(1:size(C, 1));
Ctext = num2str(C(:));
text(Xgrid(:), Ygrid(:), Ctext)
labels = data_matrix(1,:);
set(gca, 'XTick', 1:size(C, 1), 'XTickLabel', labels, ...
         'YTick', 1:size(C, 1), 'YTickLabel', labels, ...
         'XTickLabelRotation', 30, ...
         'TickLabelInterpreter', 'none')
xlabel('Predicted Class')
ylabel('Known Class')
title('Forest Confusion Matrix')
toc
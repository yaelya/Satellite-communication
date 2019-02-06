tic
clc; clear all
data_matrix=xlsread('file1.xlsx');
icol = size(data_matrix,2)

data_predictor = data_matrix(:,1:icol-1); % predictors matrix
label = data_matrix(:,end); % last column is 2 for benign, 4 for malignant


cvp=cvpartition(length(data_predictor),'holdout',0.33); 
%Training set
Xtrain=data_predictor(training(cvp),:);
Ytrain=label(training(cvp),:);

%Testing set
Xtest=data_predictor(test(cvp),:);
Ytest=label(test(cvp),:);

BaggedEnsemble = generic_random_forests(Xtrain, Ytrain, 500, 'classification')
BaggedEnsemble = generic_random_forests(Xtest, Ytest, 60, 'classification')

%%predict(BaggedEnsemble, [5 3 5 1.8])

% Model says that x6 (single epithelial cell size) is most important
% predictor
toc

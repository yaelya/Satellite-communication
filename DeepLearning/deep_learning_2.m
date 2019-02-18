tic
clc; clear all
data=xlsread('file1.xlsx');
data_predictor = data(:,1:end-1); % predictors matrix
label = data(:,end); % last column is 2 for benign, 4 for malignant


cvp = cvpartition(length(data_predictor),'Holdout',0.3);
dataTrain = data(training(cvp),:);
dataHeldOut = data(test(cvp),:);




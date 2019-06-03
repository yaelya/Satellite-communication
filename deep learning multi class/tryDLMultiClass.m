tic
clc; clear all
% data=xlsread('34_35_1.xlsx');
% data(:,1:end-1)=zscore(data(:,1:end-1));
% 
% [train,test] = holdout(data,80);
% Test set
% Xtest=test(:,1:end-1);Ytest=test(:,end);
% Traing set
% XTrain=train(:,1:end-1);YTrain=train(:,end);

[XTrain,YTrain] = japaneseVowelsTrainData;
XTrain(1:5)

figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')
toc

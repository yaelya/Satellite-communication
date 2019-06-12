%https://www.mathworks.com/help/signal/examples/classify-ecg-signals-using-long-short-term-memory-networks.html
tic
%clc; clear all
data=xlsread('NIRMUL_ABS.xlsx');
Y = data(:,end);
X = data(:,1:end-1);
part = cvpartition(Y,'Holdout',20);
istrain = training(part); % Data for fitting
istest = test(part);      % Data for quality assessment

mu = mean(istrain);
sig = std(istrain);

dataTrainStandardized = (istrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;
net = trainNetwork(XTrain,YTrain);


toc
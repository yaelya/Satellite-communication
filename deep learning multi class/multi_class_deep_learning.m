%https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html
%https://www.mathworks.com/help/deeplearning/examples/classify-sequence-data-using-lstm-networks.html#ClassifySequenceDataUsingLSTMNetworksExample-4

tic
clc; clear all
data=xlsread('34_35_1.xlsx');

%disp(length(data));
data(:,1:end-1)=zscore(data(:,1:end-1));
[train,test] = holdout(data,80);
% Test set
Xtest=test(:,1:end-1);Ytest=test(:,end);
% Traing set
XTrain=train(:,1:end-1);YTrain=train(:,end);

figure
plot(XTrain)
title("Training Observation 1")
numFeatures = size(XTrain,1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')


inputSize = 12;
numHiddenUnits = 100;
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XTest,'MiniBatchSize',miniBatchSize);

acc = sum(YPred == YTest)./numel(YTest)


toc
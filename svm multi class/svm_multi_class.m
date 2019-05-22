%https://www.mathworks.com/help/stats/classificationecoc.html
%clc; clear all
data=xlsread('34_35.xlsx');

%disp(length(data));
data(:,1:end-1)=zscore(data(:,1:end-1));
[train,test] = holdout(data,80);
% Test set
Xtest=test(:,1:end-1);Ytest=test(:,end);
% Traing set
XTrain=train(:,1:end-1);YTrain=train(:,end);

x=multisvm(XTrain,YTrain,Xtest);

toc
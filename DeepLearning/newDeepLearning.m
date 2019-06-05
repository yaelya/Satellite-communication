%https://www.mathworks.com/help/signal/examples/classify-ecg-signals-using-long-short-term-memory-networks.html
tic
%clc; clear all
data=xlsread('NIRMUL_ABS.xlsx');
Y = data(:,end);
X = data(:,1:end-1);

summary(X)
toc
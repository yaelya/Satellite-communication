%https://www.mathworks.com/help/signal/examples/classify-ecg-signals-using-long-short-term-memory-networks.html

tic
%clc; clear all
data=xlsread('34_35_1.xlsx');

[Signals,Labels] = segmentSignals(Signals,Labels);


toc
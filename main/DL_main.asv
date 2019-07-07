function [Yfit]=DL_main(data)
%https://www.mathworks.com/help/stats/classification-with-imbalanced-data.html
tic
Y = data(:,end);
X = data(:,1:end-1);

rng(10,'twister');         % For reproducibility
part = cvpartition(Y,'holdout',0.2);
istrain = training(part); % Data for fitting
istest = test(part);      % Data for quality assessment
%tabulate(Y(istrain));

N = sum(istrain);         % Number of observations in the training sample
t = templateTree('MaxNumSplits',N);
rusTree = fitcensemble(data(istrain,1:11),Y(istrain),'Method','RUSBoost', ...
    'NumLearningCycles',150,'Learners',t,'LearnRate',0.1,'nprint',100);
figure;
plot(loss(rusTree,data(istest,1:11),Y(istest),'mode','cumulative'));
grid on;
xlabel('Number of trees');
ylabel('Test classification error');

Yfit = predict(rusTree,data(istrain,1:11));
tab = tabulate(Y(istrain));
mat = bsxfun(@rdivide,confusionmat(Y(istrain),Yfit),tab(:,2));


cmpctRus = compact(rusTree);

sz(1) = whos('rusTree');
sz(2) = whos('cmpctRus');
[sz(1).bytes sz(2).bytes];
cmpctRus = compact(rusTree);

sz(1) = whos('rusTree');
sz(2) = whos('cmpctRus');
[sz(1).bytes sz(2).bytes];

L=loss(cmpctRus,data(istrain,1:11),Y(istrain));

%X = sprintf('accuracy: %f\n', mean(diag(cmpctRus)));
%disp(X)?
%X

%clear data;

toc
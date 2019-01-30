%https://www.mathworks.com/help/stats/classification-with-imbalanced-data.html
tic
data=xlsread('file.xlsx');
Y = data(:,end);
data(:,end) = [];
tabulate(Y)

rng(10,'twister')         % For reproducibility
part = cvpartition(Y,'Holdout',0.5);
istrain = training(part,1); % Data for fitting
%istest = test(part);      % Data for quality assessment
tabulate(Y(istrain))

N = sum(istrain);         % Number of observations in the training sample
t = templateTree('MaxNumSplits',N);
tic
rusTree = fitcensemble(data(istrain,:),Y(istrain),'Method','RUSBoost', ...
    'NumLearningCycles',1000,'Learners',t,'LearnRate',0.1,'nprint',100);

figure;
tic
plot(loss(rusTree,data(istrain,:),Y(istrain),'mode','cumulative'));
toc
grid on;
xlabel('Number of trees');
ylabel('Test classification error');

Yfit = predict(rusTree,data(istrain,:));
tab = tabulate(Y(istrain));
bsxfun(@rdivide,confusionmat(Y(istrain),Yfit),tab(:,2))*100

cmpctRus = compact(rusTree);

sz(1) = whos('rusTree');
sz(2) = whos('cmpctRus');
[sz(1).bytes sz(2).bytes]
cmpctRus = compact(rusTree);

sz(1) = whos('rusTree');
sz(2) = whos('cmpctRus');
[sz(1).bytes sz(2).bytes]

L = loss(cmpctRus,data(istrain,:),Y(istrain))
toc

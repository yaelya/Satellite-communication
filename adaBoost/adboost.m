% https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat
%% from the site belowe

tic
clc; clear all

[~, ~, ctg] = xlsread('file1.xlsx');
X = cell2mat(ctg(2:end, 1:end-1));
y = cell2mat(ctg(2:end, end));
colLabel = ctg(1, 1:end-1);

verbose = true;
classifier = AdaBoost_mult(decision_stump, verbose); % blank classifier
nTree = 12;
C = classifier.train(X, y, [], nTree);


[accuracy, tpr] = C.calc_accuracy( X, y, nTree);


hold on
plot(tpr)
plot(accuracy, 'LineWidth',2)
xlabel('Number of iterations')
legend({'true positive rate of normal patients', 'true positive rate of suspect patients', ...
  'true positive rate of Pathologic patients', 'overall accuracy'},'Location','southeast')
accuracy
[hx, hy] = C.feature_hist();
hy = 100*hy/sum(hy);  % normalize and convert to percent
hx(hy==0) = [];       % delete unused features
hy(hy==0) = [];
[hy, idx] = sort(hy); % sort
hx = hx(idx);
clf;
barh(hy);
axis tight;
xlabel('Percent used');
ylabel('Feature name');
ax = gca;
ax.YTick = 1:length(hx);
ax.YTickLabel = colLabel(hx);

[strList, labels, header] = C.export_model();
CC = classifier.import_model(strList, labels); % initialize new model
Y  = C.predict(X);
YY = CC.predict(X);
% fprintf('Number of mismatches between models: %i\n', nnz(Y~=YY));
 
save_adaboost_model(C, 'classifier.csv');
CC = load_adaboost_model(classifier, 'classifier.csv');
Y  = C .predict(X);
YY = CC.predict(X);
% % fprintf('Number of mismatches between models: %i\n', nnz(Y~=YY));
% type('classifier.csv');

% fprintf('Classification is %i%% accurate when training and testing on the same data.\n', ...
%  round(100*mean(y==Y)));
classifier = AdaBoost_mult(decision_stump);
nFold = 10; % ten-fold validation
%Y = cross_validation( classifier, X, y, nTree, nFold);
% fprintf('Classification is %i%% accurate when using 10-fold cross validation\n', ...
%    round(100*mean(y==Y)));


% nSamp = 1000;
% [Xb,Yb] = pol2cart((1:nSamp)*2*pi/nSamp,3);
% X = 2*[randn(nSamp,2); randn(nSamp,2)+ [Xb' ,Yb'] ];
% y = [1+zeros(nSamp,1); 2+zeros(nSamp,1)];
% nTree   = 30;   % number of trees
% C = classifier.train(X, y, [], nTree);
% AdaBoost_demo_plot(C, X, y);

% nSamp = 1000;
% [Xb,Yb] = pol2cart((1:nSamp)*2*pi/nSamp,3);
% [Xc,Yc] = pol2cart((1:nSamp)*2*pi/nSamp,6);
% X= 1.25*[randn(nSamp,2); randn(nSamp,2)+[Xb',Yb']; randn(nSamp,2)+[Xc',Yc'] ];
% y = [1+zeros(nSamp,1); 2+zeros(nSamp,1); 3+zeros(nSamp,1)];
% nTree = 30;   % number of trees
% C = classifier.train(X, y, [], nTree);
% Y = C.predict(X);
% AdaBoost_demo_plot(C, X, y);


% nSamp = 1000;
% Xa = (1:nSamp)*10/nSamp;
% d = 6;
% s = sign(randn(nSamp,1));
% X = [randn(nSamp,2)+[Xa',Xa']; randn(nSamp,2)+[Xa',Xa'+s.*d]]-5;
% y = [1+zeros(nSamp,1); 2+zeros(nSamp,1)];
% nTree   = 30;   % number of trees
% C = classifier.train(X, y, [], nTree);
% AdaBoost_demo_plot(C, X, y);

nSamp = 1000;
Xa = (1:nSamp)*10/nSamp;
d = 4;
X = [randn(nSamp,2)+[Xa',Xa']; randn(nSamp,2)+[Xa',Xa'+d]-1; randn(nSamp,2)+[Xa',Xa'-d]+1]-5;
y = [1+zeros(nSamp,1); 2+zeros(nSamp,1); 3+zeros(nSamp,1)];
nTree = 30;   % number of trees
C = classifier.train(X, y, [], nTree);
AdaBoost_demo_plot(C, X, y);


toc
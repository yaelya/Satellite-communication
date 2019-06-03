%   Gentle AdaBoost Classifier with two different weak-learners : Decision Stump and Perceptron.
%   Multi-class problem is performed with the one-vs-all strategy.
%
%   Usage
%   ------
%
%   model = gentleboost_model(X , y , [options]);
%
%
%   Inputs
%   -------
%
%   X                                     Features matrix (d x N) in double precision
%   y                                     Labels vector(1 x N) where y_i={1,...,M} and i=1,...,N.. If y represent binary labels vector then y_i={-1,1}.
%   options
%               weaklearner               Choice of the weak learner used in the training phase
%                                         weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
%                                         weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
%               T                         Number of weaklearners (default T = 100)
%               epsi                      Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1.0)
%               lambda                    Regularization parameter for the perceptron's weights update (default lambda = 1e-3)
%               max_ite                   Maximum number of iterations of the perceptron algorithm (default max_ite = 100)
%               seed                      Seed number for internal random generator (default random seed according to time)
%
% If compiled with the "OMP" compilation flag
%              num_threads                Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)
%
%   Outputs
%   -------
%
%   model                                 Structure of model ouput
%
%               featureIdx                Features index in single/double precision of the T best weaklearners (T x m) where m is the number of class.
%                                         For binary classification m is force to 1.
%               th                        Optimal Threshold parameters (1 x T) in single/double precision.
%               a                         Affine parameter(1 x T) in single/double precision.
%               b                         Bias parameter (1 x T) in single/double precision.
%               weaklearner               Choice of the weak learner used in the training phase in single/double precision.
%               epsi                      Epsilon constant in the sigmoid function used in the perceptron in single/double precision.

clear, clc, close all,drawnow

load iris

labels               = unique(y);

options.method       = 7;
options.holding.rho  = 0.7;
options.holding.K    = 50;

options.weaklearner  = 0;
options.epsi         = 0.1;
options.lambda       = 1e-2;
options.max_ite      = 1000;
options.T            = 5;

positive             = labels(3);
ind_positive         = find(labels==positive);

[d , N]              = size(X);
[Itrain , Itest]     = sampling(X , y , options);

[Ncv , Ntrain]       = size(Itrain);
Ntest                = size(Itest , 2);
error_train          = zeros(1 ,  Ncv);

error_test           = zeros(1 ,  Ncv);

tptrain              = zeros(Ncv , 100);
fptrain              = zeros(Ncv , 100);

tptest               = zeros(Ncv , 100);
fptest               = zeros(Ncv , 100);

for i=1:Ncv

    [Xtrain , ytrain , Xtest , ytest]  = samplingset(X , y , Itrain , Itest , i);

    model_gentle                       = gentleboost_model(Xtrain , ytrain , options);

    [ytrain_est , fxtrain]             = gentleboost_predict(Xtrain , model_gentle);
    error_train(i)                     = sum(ytrain_est~=ytrain)/Ntrain;

    ytrain(ytrain ~=positive)          = -1;
    ytrain(ytrain ==positive)          = 1;
    [tptrain(i , :) , fptrain(i , :)]  = basicroc(ytrain , fxtrain(ind_positive , :));


    [ytest_est , fxtest]               = gentleboost_predict(Xtest , model_gentle);
    error_test(i)                      = sum(ytest_est~=ytest)/Ntest;

    ytest(ytest ~=positive)            = -1;
    ytest(ytest ==positive)            = 1;
    [tptest(i , :) , fptest(i , :)]    = basicroc(ytest , fxtest(ind_positive , :));

    fprintf('%d/%d\n' , i , options.holding.K)
    drawnow

end

fprintf('Error Train/Test %2.4f/%2.4f\n' , mean(error_train,2) , mean(error_test,2))

fptrain_mean                          = mean(fptrain);
tptrain_mean                          = mean(tptrain);
auc_train                             = auroc(tptrain_mean', fptrain_mean');

fptest_mean                           = mean(fptest);
tptest_mean                           = mean(tptest);
auc_test                              = auroc(tptest_mean', fptest_mean');

figure(1)

plot(fptrain_mean , tptrain_mean , 'k' , fptest_mean , tptest_mean , 'r' , 'linewidth' , 2)
%axis([-0.02 , 1.02 , -0.02 , 1.02])
axis([-0.02 , 0.2 , 0.7 , 1.02])

legend(sprintf('Train, AUC = %5.4f' , auc_train) , sprintf('Test, AUC = %5.4f' , auc_test) , 'location' , 'southeast')
xlabel('False Positive Rate' , 'fontsize' , 12 , 'fontweight' , 'bold')
ylabel('True Positive Rate' , 'fontsize' , 12 , 'fontweight' , 'bold')
title(sprintf('Gentleboost/weak = %d, d = %d, positive = %d, T = %d' , options.weaklearner , d , positive , options.T) , 'fontsize' , 13 , 'fontweight' , 'bold')
grid on
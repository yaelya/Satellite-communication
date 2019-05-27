%https://www.mathworks.com/matlabcentral/fileexchange/22997-multiclass-gentleadaboosting
[~, ~, ctg] = xlsread('34_35_1.xlsx');
X = cell2mat(ctg(2:end, 1:end-1));
y = cell2mat(ctg(2:end, end));

options.method       = 4;
options.holding.rho  = 0.7;
options.holding.K    = 50;

options.weaklearner  = 0;
options.epsi         = 0.1;
options.lambda       = 1e-2;
options.max_ite      = 1000;


Tmin                 = 1;
stepT                = 2;
Tmax                 = 50;
T                    = (Tmin:stepT:Tmax);

NT                   = length(T);

[d , N]              = size(X);
[Itrain , Itest]     = sampling(X , y , options);

[Ncv , Ntrain]       = size(Itrain);
Ntest                = size(Itest , 2);

error_train          = zeros(NT ,  Ncv);
error_test           = zeros(NT ,  Ncv);


for t = 1:NT

    options.T  = T(t);
    for i=1:Ncv

        [Xtrain , ytrain , Xtest , ytest]  = samplingset(X , y , Itrain , Itest , i);

        model_gentle = gentleboost_model(Xtrain , ytrain , options);
        [ytrain_est , fxtrain]= gentleboost_predict(Xtrain , model_gentle);
        error_train(t,i)= sum(ytrain_est~=ytrain)/Ntrain;

        [ytest_est , fxtest]= gentleboost_predict(Xtest , model_gentle);
        error_test(t,i)= sum(ytest_est~=ytest)/Ntest;

    end
end

error_train_T = mean(error_train , 2);
error_test_T  = mean(error_test , 2);
[mini , pos]  = min(error_test_T);



figure(2)
plot(T , error_train_T , T , error_test_T  , 'r' , T(pos) , mini , 'ko'  , 'linewidth' , 2)
axis([0.5 , Tmax+0.25 , -0.01 , 1.1*max(max(error_train_T),max(error_test_T))])

xlabel('Number of weak-learners' , 'fontsize' , 12 , 'fontweight' , 'bold')
ylabel('Error rate' , 'fontsize' , 12 , 'fontweight' , 'bold')
title(sprintf('Gentleboost/weak = %d, d = %d' , options.weaklearner , d ) , 'fontsize' , 13 , 'fontweight' , 'bold')
legend('Train' , 'Test')
grid
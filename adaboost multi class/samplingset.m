function [Xtrain , ytrain , Xtest , ytest , Xvalid , yvalid] = samplingset(X , y , Itrain , Itest , i , Ivalid);
% [Xtrain , ytrain , Xtest , ytest ,  [Xvalid] , [yvalid]] = samplingset(X , y , i , Itrain , Itest , [i] , [Ivalid]);
%
% 
% example
% -------
% load wine
% sigma               = 0.4;
% options.method      = 7;
% options.holding.rho = 0.7;
% options.holding.K   = 100;
% 
% [Itrain , Itest]    = sampling(X , y , options);
% nbite               = size(Itrain , 1);
% Perf                = zeros(1 , nbite);
% 
% for i=1:nbite
% 
%    [Xtrain , ytrain , Xtest , ytest] = samplingset(X , y , Itrain , Itest , i);
%    ytest_est                         = parzen_classif(Xtrain , ytrain , Xtest , sigma );
%    Perf(i)                           = perf_classif(ytest , ytest_est);
% 
% end
% 
% disp(mean(Perf))
%
% Author : S?bastien PARIS : sebastien.paris@lsis.org
% -------
if (nargin < 5)
    
    i    = ceil(size(Itrain , 1)*rand);
    
end
if (nargin < 6)
    
    Xvalid = [];
    
    yvalid = [];
    
end
if ((i < 1) || (i > size(Itrain , 1)))
    
    error('index i is not valid');
    
end
indtrain    = Itrain(i , :);
Xtrain      = X(: , indtrain);
ytrain      = y(indtrain);
indtest     = Itest(i , :);
Xtest       = X(: , indtest);
ytest       = y(indtest);
if (nargin == 6)
    indvalid     = Ivalid(i , :);
    Xvalid       = X(: , indvalid);
    yvalid       = y(indvalid);
end

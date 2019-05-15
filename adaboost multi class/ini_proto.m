function [yproto , Wproto , lambda] = ini_proto(Xtrain , ytrain , Nproto_pclass , option)
% Initialize Prototypes Weights and class label
%
% Usage
% ------
%
% [yproto , Wproto , lambda] = ini_proto(Xtrain , ytrain , [Nproto_pclass] , [option])
%
% Inputs
% -------
%
% Xtrain            Train data (d x Ntrain)
% ytrain            Labels (1 x Ntrain), card(ytrain) = m
% Nproto_pclass     Number of prototype per class (1 x m)
%                   Nproto = sum(Nproto_pclass)
% option            =1 Wproto ~N(E[Xtrain|y=i] , Cov[Xtrain|y=i]); 2 = random vector from Xtrain
%
%
% Outputs
% -------
%
% yproto            Prototype labels  (1 x Nproto)
% Wproto            Prototype Weights (d x Nproto)
% lambda            Lambda values
%
%  Author : S?bastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 04/09/2006
if (nargin < 4)
    option = 1;
end
[d , Ntrain] = size(Xtrain);
labels       = unique(ytrain);
m            = length(labels);
if (nargin < 3)
    Nproto_pclass = round(sqrt(Ntrain))*ones(1 , m);
end
if(size(Nproto_pclass , 2) ~= m)
    error('Nproto_pclass must be (1 x m) vector');
end
Nproto       = sum(Nproto_pclass);
Wproto       = zeros(d , Nproto);
yproto       = zeros(1 , Nproto);
co           = 1;
for i = 1 : m
    ind                             = (co:co + Nproto_pclass(i) - 1);
    indice                          = find(ytrain == labels(i));
    if (option == 1)
        
        Xi                              = Xtrain(: , indice);
        Ni                              = size(Xi , 2);
        EXi                             = sum(Xi , 2)/Ni;
        res                             = (Xi - EXi(: , ones(1 , Ni)));
        
        %    CovXi                           = res*res'/(Ni - 1);
        Sigmai                          = sqrt(sum(res.*res , 2)/(Ni - 1));
        ONproto                         = ones(1 , Nproto_pclass(i));
        
        %    Wproto(: , ind)                 = EXi(: , ONproto) + %    chol(CovXi)'*randn(d , Nproto_pclass(i));
        Wproto(: , ind)                 = EXi(: , ONproto) + Sigmai(: , ONproto).*randn(d , Nproto_pclass(i));
    else
        Ni                              = length(indice);
        tempindice                      = randperm(Ni);
        r                               = indice(tempindice(1:Nproto_pclass(i)));
        Wproto(: , ind)                 = Xtrain(: , r);
        
    end
    yproto(ind)                     = labels(i);
    co                              = co + Nproto_pclass(i);
    
end
lambda  = ones(d , 1)/d;
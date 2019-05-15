function [Itrain , Itest , Ivalid] = sampling(X , y , options);
% Various Data sampling methods for evaluate Classifier Performances.
% 
% X                           : data (d x N)
% y                           : labels(1 x N)
% options.valid               : 1 = 3 sets in output : train, test, valid. 0 = 2 output : train, test
% options.randomize           : 1 = do a randomize permutation before splitting set
% options.fraction            : 0<x<1 : build set only with the fraction of the entiere set (for huge dataset)
% options.maxperclass         : Max samples per class. A scalar or a (1 x m) vector
% options.method              : 1 = Hold out
%                               2 = Bootstrap
%                               3 = K Cross-validation
%                               4 = Leave One Out
%                               5 = Stratified Cross Validation
%                               6 = Balanced Stratified Cross Validation
%                               7 = Stratified Hold out
%                               8 = Stratified Boot Strap
%
%
% load iris
% [Itrain , Itest , Ivalid] = sampling(X , y);
% 
% Author : S?bastien PARIS : sebastien.paris@lsis.org
% -------
if (nargin < 3)
    options.valid       = 0;
    options.randomize   = 1;
    options.fraction    = 1;
    
    options.maxperclass = inf;
    
    options.seed        = -1;
    options.method      = 3;
    options.cv.K        = 10;
else
    if (~any(strcmp(fieldnames(options) , 'valid')))
        options.valid         = 0;
    end
    if (~any(strcmp(fieldnames(options) , 'randomize')))
        options.randomize     = 1;
    end
    if (~any(strcmp(fieldnames(options) , 'fraction')))
        options.fraction      = 1;
    end
    if (~any(strcmp(fieldnames(options) , 'maxperclass')))
        options.maxperclass   = inf;
    end
    
    if (~any(strcmp(fieldnames(options) , 'seed')))
        options.seed   = -1;
    end
    
    
    
    if (~any(strcmp(fieldnames(options) , 'method')))
        options.method        = 3;
    end
    if (~any(strcmp(fieldnames(options) , 'cv.K')))
        options.cv.K         = 10;
    end
end
[d , N]               = size(X);
if(options.seed ~= -1)
    
   rand('state' , options.seed); 
    
end
if (options.randomize)
    indN              = randperm(N);
    X                 = X(: , indN);
    y                 = y(indN);
end
if ((options.fraction > 0) && (options.fraction < 1))
    
    N                = round(options.fraction*N);
    indN             = (1 : N);
    
    X                = X(: , indN);
    y                = y(indN);
end
if(any(~isinf(options.maxperclass)))
   
    label            = unique(y);
    
    L                = length(label);
    
    if(numel(options.maxperclass) == 1)
        
        options.maxperclass = options.maxperclass*ones(1 , L);
        
    end
        
    ind              = cell(1 , L);
    
    indice           = [];
    for i = 1 : L
        ind{i}       = find(y == label(i));
        
        indice       = [indice , ind{i}(1:(min(length(ind{i}) , options.maxperclass(i))))];
    end
    X                = X(: , indice);
    
    y                = y(: , indice);
    
    [d , N]          = size(X);
    
    
end
indN             = (1 : N);
Ivalid               = [];
if(options.method == 0)
   
    Itrain           = indN;
    
    Itest            = [];
    
   
end
if(options.method == 1) %holding method sans remise
    rho              = options.holding.rho;
    K                = options.holding.K;
    if options.valid
        if(prod(size(rho)) == 1)
            rho     = [rho , (1-rho)/2 , (1-rho)/2];
        end
    end
    NN               = round(rho*N);
    Ntrain           = NN(1);
    if(options.valid)
        Ntest        = NN(2);
        Nvalid       = N - (Ntrain + Ntest);
        Ivalid       = zeros(K , Nvalid);
    else
        Ntest        = N - Ntrain;
    end
    Itrain           = zeros(K , Ntrain);
    Itest            = zeros(K , Ntest);
    for i = 1 : K
        temp          = randperm(N);
        Itrain(i , :) = temp(1:Ntrain);
        if(options.valid)
            Itest(i , :)  = temp((Ntrain+1):(Ntrain + Ntest));
            Ivalid(i , :) = temp(Ntrain + Ntest + 1:N);
        else
            Itest(i , :)  = temp(Ntrain + 1:N);
        end
    end
end
if(options.method == 2) %Bootstrap method avec remise
    rho               = options.bootstraping.rho;
    K                 = options.bootstraping.K;
    if options.valid
        if(prod(size(rho)) == 1)
            rho     = [rho , (1-rho)/2 , (1-rho)/2];
        end
    end
    NN               = round(rho*N);
    Ntrain           = NN(1);
    if(options.valid)
        Ntest        = NN(2);
        Nvalid       = N - (Ntrain + Ntest);
        Ivalid       = zeros(K , Nvalid);
    else
        Ntest        = N - Ntrain;
    end
    Itrain           = zeros(K , Ntrain);
    Itest            = zeros(K , Ntest);
    for i = 1 : K
        temp          = ceil(N*rand(1 , N));
        Itrain(i , :) = temp(1:Ntrain);
        if(options.valid)
            Itest(i , :)  = temp((Ntrain+1):(Ntrain + Ntest));
            Ivalid(i , :) = temp(Ntrain + Ntest + 1:N);
        else
            Itest(i , :)  = temp(Ntrain + 1:N);
        end
    end
end
if(options.method == 3) %K Fold Cross-Validation
    K      = options.cv.K;
    S      = floor(N/K);
    indN   = (1:N);
    Itrain = zeros(K , N - S - options.valid*S);
    Itest  = zeros(K , S);
    if (options.valid)
        Ivalid = zeros(K , S);
    end
    for i = 1 : K
        Itest(i , :)  = indN((i-1)*S+1:i*S);
        temp          = indN([i*S+1:N , 1:(i-1)*S]);
        if(options.valid)
            Ivalid(i , :) = temp(1:S);
            Itrain(i , :) = temp(S+1:end);
        else
            Itrain(i , :) = temp;
        end
    end
end
if(options.method == 4) % Leave One Out
    K      = N;
    S      = 1;
    indN   = (1:N);
    Itrain = zeros(K , N - S - options.valid*S);
    Itest  = zeros(K , S);
    if (options.valid)
        Ivalid = zeros(K , S);
    end
    for i = 1 : K
        Itest(i , :)  = indN((i-1)*S+1:i*S);
        temp          = indN([i*S+1:N , 1:(i-1)*S]);
        if(options.valid)
            Ivalid(i , :) = temp(1:S);
            Itrain(i , :) = temp(S+1:end);
        else
            Itrain(i , :) = temp;
        end
    end
end
if(options.method == 5) % Stratified Cross Validation
    K                              = options.cv.K;
    label                          = unique(y);
    select                         = histc(y , label);
    L                               = length(label);
    ind                             = cell(1 , L);
    n                               = zeros(1 , L);
    for i = 1 : L
        ind{i}                      = find(y == label(i));
    end
    Ntestc                          = floor((1/K).*select);
    if(options.valid)
        Nvalidc                       = Ntestc;
        Ntrainc                       = select  - 2*Ntestc;
        Ntrain                        = sum(Ntrainc);
        Ntest                         = sum(Ntestc);
        Nvalid                        = sum(Nvalidc);
        Ivalid                        = zeros(K , Nvalid);
    else
        Ntrainc                         = select  - Ntestc;
        Ntrain                          = sum(Ntrainc);
        Ntest                           = sum(Ntestc);
    end
    Itrain                          = zeros(K , Ntrain);
    Itest                           = zeros(K , Ntest);
    for j = 1 : K
        temptrain                   = [];
        temptest                    = [];
        tempvalid                   = [];
        for ii = 1 : L
            temptest                = [temptest , ind{ii}(1+(j-1)*Ntestc(ii):j*Ntestc(ii))];
            temp                    = [1+j*Ntestc(ii): select(ii) , 1:(j-1)*Ntestc(ii) ];
            if(options.valid)
                tempvalid               = [tempvalid , ind{ii}(temp(1:Nvalidc(ii)))];
                temptrain               = [temptrain , ind{ii}(temp(Nvalidc(ii)+1:end))];
            else
                temptrain               = [temptrain , ind{ii}(temp)];
            end
        end
        Itest(j , :)                   = temptest;
        Itrain(j , :)                  = temptrain;
        if(options.valid)
            Ivalid(j , :)                   = tempvalid;
        end
    end
end
if(options.method == 6) % Balanced Stratified Cross Validation
    K                              = options.cv.K;
    label                          = unique(y);
    select                         = histc(y , label);
    [d , N]                        = size(X);
    L                               = size(label , 2);
    ind                             = cell(1 , L);
    list                            = cell(1 , L);
    ind_list                        = cell(1 , L);
    Ni                              = zeros(1 , L);
    Ntestc                          = floor((1/K).*select);
    if(options.valid)
        Nvalidc                       = Ntestc;
        Ntrainc                       = select  - 2*Ntestc;
        Ntrain                        = sum(Ntrainc);
        Ntest                         = sum(Ntestc);
        Nvalid                        = sum(Nvalidc);
        Ivalid                        = zeros(K , Nvalid);
    else
        Ntrainc                         = select  - Ntestc;
        Ntrain                          = sum(Ntrainc);
        Ntest                           = sum(Ntestc);
    end
    Itrain                          = zeros(K , Ntrain);
    Itest                           = zeros(K , Ntest);
    for i = 1 : L
        ind{i}                      = find(y == label(i));
        Ni(i)                       = length(ind{i});
        list{i}                     = zeros(d , Ni(i) + 1);
        ind_list{i}                 = zeros(1 , Ni(i));
    end
    temp                            = 0;
    for i = 1 : L
        temp                             = temp + 1;
        a                                = X(: , ind{i});
        list{i}(: , 1)                   = min(a , [] , 2);
        for j=1:Ni(i)
            b                            = list{i}(: , j);
            tmp                          = b(: , ones(1 , size(a , 2))) - a;
            dist                         = sqrt(sum(tmp.*tmp));
            [minval , s]                 = min(dist);
            list{i}(: , j + 1)           = a(: , s);
            ind_list{i}(j)               = ind{i}(s);
            a(: , s)                     = [];
            ind{i}(s)                    = [];
        end
    end
    T_ind                                = cell(1 , K);
    Listremain                           = [];
    for i=1:temp
        b_ind                        = ind_list{i} ;
        if ~isempty(b_ind)
            while (size(b_ind,2) >= K)
                for j=1:K
                    T_ind{j}         = [T_ind{j} ; b_ind(1)];
                    b_ind(1)         = [];
                end
            end
            Listremain               = [Listremain , b_ind];
            b_ind                    = [];
        end
    end
    %     while ~isempty(Listremain)
    %         for i=1:K
    %             if ~isempty(Listremain)
    %
    %                 T_ind{i}              = [T_ind{i} ; Listremain(1)];
    %
    %                 Listremain(1)        = [];
    %             end
    %         end
    %     end
    %     Ntest                           = length(T_ind{1});
    %
    %     Ntrain                          = (K - 1)*Ntest + length(Listremain);
    Iindice                          = options.valid.*[[zeros(K-1,1) ; 1] , eye(K,K-1) ] + eye(K);
    for i=1:K
        tmp                          = Iindice(i , :)==1;
        indtest                      = find(tmp);
        indtrain                     = find(~tmp);
        Itrain_temp                  = [];
        for j=1:length(indtrain)
            Itrain_temp              = [Itrain_temp ; T_ind{indtrain(j)}];
        end
        [ignore,p]                   = sort(rand(1 , Ntrain));
        temp                         = [Itrain_temp' , Listremain];
        Itrain(i , :)                = temp(p);
        Itest(i , :)                 = T_ind{indtest(1)}';
        if(options.valid)
            Ivalid(i , :)             = T_ind{indtest(2)}';
        end
    end
end
if(options.method == 7) % Stratified Hold out
    rho              = options.holding.rho;
    K                = options.holding.K;
    if options.valid
        if(prod(size(rho)) == 1)
            rho     = [rho , (1-rho)/2 , (1-rho)/2];
        end
    end
    label            = unique(y);
    select           = histc(y , label);
    L                = size(label , 2);
    ind              = cell(1 , L);
    for i = 1 : L
        ind{i}  = find(y == label(i));
    end
    Ntrainc          = round(rho(1)*select);
    Ntrain           = sum(Ntrainc);
    if(options.valid)
        Ntestc       = round(rho(2)*select);
        Ntest        = sum(Ntestc);
        Nvalidc      = select - Ntrainc - Ntestc;
        Nvalid       = sum(Nvalidc);
        Ivalid       = zeros(K , Nvalid);
    else
        Ntestc           = select - Ntrainc;
        Ntest            = sum(Ntestc);
    end
    Itrain           = zeros(K , Ntrain);
    Itest            = zeros(K , Ntest);
    for j = 1 : K
        temptrain        = [];
        temptest         = [];
        tempvalid        = [];
        for i = 1 : L
            [ignore , pi] = sort(rand(1 , select(i)));
            temptrain     = [temptrain , ind{i}(pi(1:Ntrainc(i)))];
            if(options.valid)
                temptest       = [temptest , ind{i}(pi(Ntrainc(i)+1:Ntrainc(i) + Ntestc(i)))];
                tempvalid      = [tempvalid , ind{i}(pi(Ntrainc(i) + Ntestc(i)+1:select(i)))];
            else
                temptest      = [temptest , ind{i}(pi(Ntrainc(i)+1:select(i)))];
            end
        end
        Itrain(j , :)    = temptrain;
        Itest(j , :)     = temptest;
        if(options.valid)
            Ivalid(j , :) = tempvalid;
        end
    end
end
if(options.method == 8) % Stratified Boot Strap
    rho              = options.bootstraping.rho;
    K                = options.bootstraping.K;
    if options.valid
        if(prod(size(rho)) == 1)
            rho     = [rho , (1-rho)/2 , (1-rho)/2];
        end
    end
    label            = unique(y);
    select           = histc(y , label);
    L                = size(label , 2);
    ind              = cell(1 , L);
    for i = 1 : L
        ind{i}  = find(y == label(i));
    end
    Ntrainc          = round(rho(1)*select);
    Ntrain           = sum(Ntrainc);
    if(options.valid)
        Ntestc       = round(rho(2)*select);
        Ntest        = sum(Ntestc);
        Nvalidc      = select - Ntrainc - Ntestc;
        Nvalid       = sum(Nvalidc);
        Ivalid       = zeros(K , Nvalid);
    else
        Ntestc           = select - Ntrainc;
        Ntest            = sum(Ntestc);
    end
    Itrain           = zeros(K , Ntrain);
    Itest            = zeros(K , Ntest);
    for j = 1 : K
        temptrain   = [];
        temptest    = [];
        tempvalid   = [];
        for i = 1 : L
            pi            = ceil(select(i)*rand(1 , select(i)));
            temptrain     = [temptrain , ind{i}(pi(1:Ntrainc(i)))];
            if(options.valid)
                temptest       = [temptest , ind{i}(pi(Ntrainc(i)+1:Ntrainc(i) + Ntestc(i)))];
                tempvalid      = [tempvalid , ind{i}(pi(Ntrainc(i) + Ntestc(i)+1:select(i)))];
            else
                temptest      = [temptest , ind{i}(pi(Ntrainc(i)+1:select(i)))];
            end
        end
        Itrain(j , :)    = temptrain;
        Itest(j , :)     = temptest;
        if(options.valid)
            Ivalid(j , :) = tempvalid;
        end
    end
end

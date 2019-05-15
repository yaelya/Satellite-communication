function [tp , fp , threshold] = basicroc(y , f , options)
%
%[tp , fp , threshold] = basicroc(y , f , [positive]);
%
% Return the true positive rate, the false positive rate and the threshold
%
%
%  Author : S?bastien PARIS : sebastien.paris@lsis.org
%  -------  Date : 04/09/2006
if nargin < 3
    
    options.positive = 1;
    
    options.nbstep   = 100;
    
end
if (~any(strcmp(fieldnames(options) , 'positive')))
    options.positive = 1;
end
if (~any(strcmp(fieldnames(options) , 'nbstep')))
    options.nbstep   = 100;
end
yunique            = unique(y);
%bin                = histc(y , yunique);
%petite modification pour le cas o? une seule classe dans le y
if (length(yunique)~=1)
    bin                = histc(y , yunique);
else
    bin                = length(y);
end
index              = (yunique==options.positive);
P                  = bin(index); %(y = positive)
if isempty(P) 
    P = 1; 
end
N                  = sum(bin(~index)); %(y = else)
minf               = min(f);
maxf               = max(f);
stepf              = (maxf - minf)/(options.nbstep - 3);
threshold          = (maxf:-stepf:minf);
tp                 = zeros(1 , options.nbstep - 2);
fp                 = zeros(1 , options.nbstep - 2);
co                 = 1;
for t = threshold
    ind       = (y(f>threshold(co))==options.positive);
    %tp(co)    = sum(ind)/(P+~isempty(P));
    %fp(co)    = sum(~ind)/(N+~isempty(N));
    tp(co)    = sum(ind)/(P);
    fp(co)    = sum(~ind)/(N);
    co        = co + 1;
end
tp                 = [0 , tp , 1];
fp                 = [0 , fp , 1];
function [threshold, dimension, labels, min_error]  = train_stump_N(X, y, w, Thr, Idx, rowMsk)
% Train Decision stump based on multi-class labes. This is low level 
% function which is used by other more user friendly classes like "decision_stump" and
% "two_level_decision_tree". Function returns best dimension and split
% thershold that optimizes error = sum(w(y~=Y)), where Y are predicted labels
%
% INPUT:
%  X - (N X D) train data set, each of N rows is a training
%    sample in the D dimensional feature space.
%  y - (N X 1) label for each entry. Has to be 1 or 2
%  w - (N x 1) weights for each entry.
%  Idx - (N X D) optional value precalculated for speed ([Thr, Idx] = sort(X,1, 'ascend'))
%  Thr - (N X D) optional value precalculated for speed ([Thr, Idx] = sort(X,1, 'ascend'))
%  rowMsk - optional mask which samples should be included (default is all)
%
% OUTPUT:
%  threshold - threshold used
%  dimension - feature number
%  labels    - two output labels
%  min_error - error sum(w(y~=Y)) where Y are predicted labels
% Based on threshold, dimension and labels, labels Y can be predicted based on X using:
%       Y = labels((X(:,dimension)>obj.threshold) + 1);
%
% PERFORMANCE:
%   Expected performance is either O(N*D) if Thr and Idx are provided or 
%   O(N*log(N)*D) if not.
%
% Written by Jarek Tuszynski, Leidos, jaroslaw.w.tuszynski_at_leidos.com
% Code released under BSD License
assert(size(X,1)==numel(y), 'Number of Rows in X has to be the same as number of elements in y.')
threshold = 0;
dimension = 1;
labels    = [0, 0];
min_error = 0;
%% defaults for optional imputs
if nargin<4
  Thr = [];
  Idx = [];
end
if nargin<6 || isempty(rowMsk)
  rowMsk = [];
  [unique_label, ~, y] = unique(y);
  max_error = sum(w);
else
  [unique_label, ~, yy] = unique(y(rowMsk));
  y(rowMsk) = yy;
  max_error = sum(w(rowMsk));
end
%% Check if the current leaf is consistent
nClass = length(unique_label); % number of unique labels or classes
if nClass < 2
  labels(:) = unique_label(1);
  return;
end
%% Check if all inputs have the same features
% We do this by seeing if there are multiple unique rows of X
% if size(unique(X(rowMsk,:),'rows'),1) == 1
%   return;
% end
if nClass==2
  [threshold, dimension, min_error]  = train_stump_2(X, y, w, Thr, Idx, rowMsk);
  labels = unique_label;
else
  C = nchoosek(unique_label,2); % all binary permutations of N labels
  thr = zeros(size(C,1),1);
  dim = thr;
  err = thr;
  for i = 1:size(C,1)
    yy = zeros(size(y));
    yy(y==C(i,1)) = 1;
    yy(y==C(i,2)) = 2;
    msk = (yy==0);
    yy(msk) = 1; % keep 2 classes: yy=1 and yy=2
    ww = w;
    ww(msk) = 0; % but set weights to other classes to 0
    [thr(i), dim(i), err(i)] = train_stump_2(X, yy, ww, Thr, Idx, rowMsk);
    if isempty(rowMsk)
      err(i) = err(i) + sum(w(msk));
    else
      err(i) = err(i) + sum(w(msk & rowMsk));
    end
  end
  [min_error, k] = min(err);
  assert(max(err)<max_error+1e-10, 'Incorrect error calculation')
  dimension = dim(k); % feature number of the dimension
  threshold = thr(k); % threshold value
  labels    = C(k,:);
end
%% reverse labels if needed
if dimension<0
  dimension = -dimension;
  labels = labels([2,1]);
end
end

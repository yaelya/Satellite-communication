function [threshold, dimension, min_error]  = train_stump_2(X, y, w, Thr, Idx, rowMsk)
% Train Decision stump based on binary labes. This is low level function 
% which is used by other more user friendly classes like "decision_stump" 
% and "two_level_decision_tree". Function returns best dimension and split
% thershold that optimizes error = sum(w(y~=Y)), where Y are predicted labels
% convert 2 labels to -1 or 1 and multiply by weight, as to
% have array with w's which are added or subtracted based on
% label
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
%  dimension - feature number times direction of the unequal sigh
%  min_error - error sum(w(y~=Y)) where Y are predicted labels
% Based on threshold, dimension and labels, labels Y can be predicted based on X using:
%       x = X(:,abs(dimension));
%       s = sign(dimension);
%       Y = (s*x>s*obj.threshold) + 1;
assert(size(X,1)==numel(y), 'Number of Rows in X has to be the same as number of elements in y.')
threshold = 0;
dimension = 0;
min_error = 0;
%% defaults for optional imputs
if nargin<4
  Thr = [];
  Idx = [];
else
  assert(all(size(X)==size(Thr)), '"Thr" has to be the same size as "X".')
  assert(all(size(X)==size(Idx)), '"Thr" has to be the same size as "X".')
end
if nargin<6 || isempty(rowMsk)
  rowMsk = [];
  unique_label = unique(y);
  sumW = sum(w);
else
  unique_label = unique(y(rowMsk));
  sumW = sum(w(rowMsk));
end
%% Check if the current leaf is consistent
nClass = length(unique_label); % number of unique labels or classes
if nClass <2
  return;
end
assert(nClass==2, 'decision_stump_2 only allows two classes.');
assert(unique_label(1)==1 && unique_label(2)==2, ...
  'decision_stump_2 only allows classes labeled 1 or 2.');
wy = w.*(2*y - 3);
if ~isempty(rowMsk) % exclude samples based on a mask
  wy(~rowMsk) = 0;
end
%% search each dimension for optimal threshold
min_error = 1;
for iFeat=1:size(X, 2)
  if isempty(Thr)
    % sort a single dimension of the data
    [thr, idx_asc] = sort(X(:,iFeat),1, 'ascend');
  else % use precomputed values
    thr = Thr(:,iFeat);
    idx_asc = Idx(:,iFeat); % ascending order
  end
  idx_dsc = idx_asc(end:-1:1);   % descending order
  if thr(1)==thr(end) && min_error<1
    continue
  end
  
  %% score each threshold. score will be between -1 and 1
  score_asc = cumsum(wy(idx_asc)); % left to right sum
  score_dsc = cumsum(wy(idx_dsc)); % right to left sum
  score = -score_asc(1:end-1) + score_dsc(end-1:-1:1);
  score(thr(1:end-1)==thr(2:end)) = 0; % if succesive threshholds are the same than skip them
  
  %% find the extreme score and weighted error
  % score=1 or -1 -> error = 0
  % score=0 -> error = 1
  [max_score, id] = max (abs (score));
  error = sumW-max_score;
  if(error < min_error)
    min_error = error;
    threshold = (thr(id) +thr(id+1) ) /2;
    dimension = sign (score (id))*iFeat; % sign * feature number
  end
end
end % train
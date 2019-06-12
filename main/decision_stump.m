classdef decision_stump
  % MULTI-CLASS THRESHOLD CLASSIFIER, a basic linear classifier where 
  % seperation hyperplane is perpedicular to one dimension.
  properties (SetAccess = private, GetAccess = private)
    dimension % feature number of the dimension 
    threshold % threshold value
    labels    % two labels
    Idx % value precalculated for speed ({obj.Thr, obj.Idx] = sort(X,1, 'ascend'))
    Thr % value precalculated for speed ([obj.Thr, obj.Idx] = sort(X,1, 'ascend'))
  end % properties
  
  methods ( Access = public )
    
    %% ===========================================================
    function obj = decision_stump(X)
      % Constructor witch can take original train data set
      if nargin>0
        % optional parameter to precompute some time consuming
        % steps
        obj = obj.preprocess_train_data(X);
      end
    end
    
    %% ===========================================================
    function obj = preprocess_train_data(obj,X)
      % precompute some time consuming steps
      if isempty(obj.Thr)
        [obj.Thr, obj.Idx] = sort(X,1, 'ascend');
      end
    end
    
    %% ===========================================================
    function obj = train(obj, X, y, w)
      % Train Decission stump in order to minimize weight of mislabeled
      % samples or sum(w(y~=Y)) where y is true and Y is predicted class
      % X - (N X D) train data set, each of N rows is a training
      %   sample in the D dimensional feature space.
      % y - (N X 1) label for each entry. Has to be 1 or 2
      % w - (N x 1) weights for each entry.
      assert(size(X,1)==numel(y), 'Number of Rows in X has to be the same as number of elements in y.')
      w = w/sum(w); % make sure weights add to 1
      [obj.threshold, obj.dimension, obj.labels] = train_stump_N(X, y, w, obj.Thr, obj.Idx);
      obj.Thr = [];                % no longer needed large data
      obj.Idx = [];                % no longer needed large data
    end % train
    
    %% ===========================================================
    function y = predict(obj, X)
      % INPUT:
      %  X - (N x [)) test data set, each of N rows is a testing
      %      sample in the D dimensional feature space.
      % OUTPUT:
      %  y - (N X 1) predicted label. Will be 1, 2 or NaN
      x = X(:,obj.dimension);
      y = obj.labels((x>obj.threshold) + 1);
      y(isnan(x)) = NaN;
      y = y(:);
    end % predict
    
    %% ===========================================================
    function str = print(obj)
      % print info about the classifier
      str = sprintf('X(%3i)>%7.3f ? %i : %i' , obj.dimension, obj.threshold, ...
        obj.labels(1), obj.labels(2));
    end % print
    
    %% ===========================================================
    function [str, dimension, header] = export_model(obj)
      % OUTPUT
      % - str     - string defining a weak classifier
      % - dimension - which dimension or feature number is this related to
      % - header  - header matching strList
      dimension = obj.dimension;
      str = sprintf('%i,%e,%i,%i', dimension, obj.threshold, obj.labels(1), obj.labels(2) );
      header = 'dimension, threshold, left label, right label';
    end
    
    %% ===========================================================
    function obj = import_model(obj,str)
      v = sscanf(str, '%i,%e,%i,%i');
      obj.dimension = v(1);
      obj.threshold = v(2);
      obj.labels    = v(3:4);
      obj.Thr = [];
      obj.Idx = [];
    end
      
    %% ===========================================================
    function t = same(obj1,obj2)
      % Compare 2 objects to see if they ar the same
      t = (obj1.dimension == obj2.dimension) && ...
          (obj1.threshold == obj2.threshold) && ...
          all(obj1.labels == obj2.labels);
    end % same
  end % methods
end % classdef

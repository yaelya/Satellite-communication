%% AdaBoostmult is an extension of AdaBoostMl from binary classification
% to multiclass classification. For example if we have 3 classes than we
% are going to break the problem down into 3 binary classifiers training on:
% 1 vs. 2+3, 2 vs. 1+3 and 3 vs. 1+2. During prediction phase we will apply
% each one of 3 classitiers to the unknown sample and pick the class with
% highest probability
classdef AdaBoost_mult
  properties (SetAccess = private, GetAccess = public)
     % weak_learner class with metchods:
    % obj = obj.train(x,y,w)
    % y = obj.predict(X)
    weak_learner % see above
  end
  properties (SetAccess = private, GetAccess = private)
    model   % struct with a model for each class
    labels  % original labels passed in the y array
    verbose = false; % print-out or not
  end % properties
  
  methods (Access = public)
    
    %% ========================================================================
    function obj = AdaBoost_mult (weak_learner_class, verbose)
      obj.weak_learner = weak_learner_class;
      if nargin>1
        obj.verbose = verbose;
      end
    end
    
    %% ========================================================================
    function [obj, accuracy] = train(obj, X, y, w, nIter)
      %train the adaboost model. For example if we have 3 classes
      % than we are going to break the problem down into 3
      % classifiers training on: 1 vs. 2+3, 2 vs. 1+3 and 3 vs. 1+2.
      % INPUT:
      % X - (N x D) train data set, each of N rows is a training
      %      sample in the D dimensional feature space.
      % y - (N x 1) true label corresponding to each sample
      % w - (N x 1) weights for each sample in case not all the
      %   points carry the same weight
      % nIter - number of iterations or weak classifiers to allow
      % OUTPUT:
      % obj - trained model
      % accuracy - (# Iterations by # classes) each column
      %   corresponds to of boolean decission accuracy per iteration
      %   of each class against all the rest.
      if isempty(w)
        w = ones(size(y));
      end
      assert(size(X,1)==numel(y), 'Number of Rows in X has to be the same as number of elements in y.')
      [obj.labels,~,y] = unique(y(:));
      obj.weak_learner = obj.weak_learner.preprocess_train_data(X);
      accuracy = [];
      for iClass = 1:length(obj.labels)
        if iscell(obj.labels)
        label = obj.labels{iClass};
        else
          label = num2str(obj.labels(iClass));
        end
        if (obj.verbose)
          fprintf('Train classifier for %s vs. the rest\n', label)
        end
        M = AdaBoost_samme(obj.weak_learner, obj.verbose);
        yy = y;
        yy(y~=iClass) = -iClass;
        [obj.model{iClass}, accuracy1] = M.train(X, yy, w, nIter);
        accuracy = [accuracy; accuracy1]; %#ok<AGROW>
      end
    end % train
    
    %% ========================================================================
    function [accuracy, tpr, classAccuracy] = calc_accuracy(obj, X, y, nIter)
      % INPUT:
      %   X - (N x D) train data set, each of N rows is a training
      %       sample in the D dimensional feature space.
      %   y - (N x 1) true label corresponding to each sample
      % OUTPUT:
      % accuracy - (# Iterations by 1) decission accuracy per iteration.
      % tpr - (# Iterations by # classes) true positive rate for each
      %   class corresponds to fraction of samples from each class which
      %   were classified correctly
      % classAccuracy - (# Iterations by # classes+l) each column
      %   corresponds to of boolean decission accuracy per iteration
      %   of each class against all the rest. The last column is
      %   overall multi-class descision accuracy or fraction of
      %   correct prediction
      [~,~,y] = unique(y(:));
      nClass = length(obj.labels);
      nIters = zeros(1,nClass);
      for iClass = 1:nClass
        nIters(iClass) = obj.model{iClass}.get_number_iterations();
      end
      if nargin<4 || isempty(nIter)
        nIter = min(nIters);
      end
      
      %% per sub_model accuracy
      nSample = length(y);
      classAccuracy = zeros (nIter,nClass)+NaN;
      P = zeros (nSample,nIter,nClass);
      for iClass = 1:nClass
        yy = y;
        yy(y~=iClass) = -iClass;
        [accuracyl, ~, prob] = obj.model{iClass}.calc_accuracy(X, yy, nIter);
        classAccuracy(:,iClass) = accuracyl;
        P(:, :,iClass) = prob(:,:,2);
      end
      
      %% overall accuracy
      nSample = size(X,1);
      [~,Y] = max(P,[],3); % pick labels for each sample and iteration
      clear P
      match = (Y==repmat(y,1,nIter));
      accuracy = sum(match,1)'/nSample;
      
      %% overall true rate
      tpr = zeros (nIter,nClass);
      for iClass = 1:nClass
        msk = (y==iClass);
        tpr(:,iClass) = sum(Y(msk, :)==iClass,1)/sum(msk);
      end
    end % end calc accuracy
    
    %% ========================================================================
    function [y, prob] = predict(obj,X,nIter)
      % Apply classifier for each class to the unknown sample and pick
      % the class with highest probability
      % INPUT:
      % X - (N x D) train data set, each of N rows is a training
      % sample in the D dimensional feature space.
      % nIter - optional parameter in case you want to try fewer
      % iterations than trained on.
      % OUTPUT:
      % y - (N x 1) predicted label corresponding to each sample
      % prob - (N by # classes> for each class probability that
      % sample belongs to class corresponding to each column and not
      % to one of other classes. Number between 0 and 1.
      if nargin<3
        nIter = [];
      end
      prob = zeros (size(X, 1),length(obj.labels));
      for iClass = 1:length(obj.labels)
        [~,p] = obj.model{iClass}.predict(X,nIter);
        prob(:,iClass) = p(:,2);
      end
      [~,id] = max(prob, [],2);
      y = obj.labels(id);
    end % predict
    
    %% ==========================================
    function [hx, hy] = feature_hist(obj)
      % which features do we use
      hy=0;
      for k = 1:length(obj.model)
        [hx, hyy] = obj.model{k}.feature_hist();
        hy = hy+hyy;
      end
    end
    
    %% ========================================================================
    function obj = optimize(obj,nIter)
      % nIter - optional parameter in case you want to try fewer
      % iterations than trained on.
      if nargin<4
        nIter = [];
      end
      for k = 1:length(obj.model)
        obj.model{k} = obj.model{k}.optimize(nIter);
      end
    end % end optimize
    
    %% ========================================================================
    
    function [strList, labels, header] = export_model(obj)
      % Output:
      % - strList - List of strings each defining a weak classifier
      % - labels  - list of labels
      % - header  - header matching strList
      strList = []; 
      labels  = [];
      for iClass = 1:length(obj.labels)
        [strList1, labels1, header] = obj.model{iClass}.export_model();
        strList = [strList; strList1]; %#ok<*AGROW>
        labels  = [labels;  labels1(:)'];
      end
    end
    
    %% ======================================================================
    function obj = import_model (obj, strList, labels)
      % Input:
      % - strList - List of strings each defining a weak classifier
      % - labels  - list of labels
        obj.labels = labels(:,2);
        nClass = size(labels,1);
        nIter  = repmat(size(strList,1)/nClass, nClass, 1);
        cnIter = cumsum(nIter);
        
        for iClass = 1:nClass
          if length(obj.model) <iClass
            obj.model{iClass} = AdaBoost_samme(obj.weak_learner);
          end
          i1 = cnIter(iClass)-nIter(iClass)+1;
          i2 = cnIter(iClass);
          obj.model{iClass} = obj.model{iClass}.import_model (strList(i1:i2, :),labels(iClass, :));
        end
    end
  end % methods
end % classdef
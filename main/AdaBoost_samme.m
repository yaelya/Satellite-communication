classdef AdaBoost_samme
  % AdaBoost SAMME algorithm described in [3]in case of binary 
  % classification it becomes AdaBoost.M1 classifier
  % so only two classes are allowed.
  % REFERENCE:
  % [1] Yoav Freund and Robert E. Shapire, ?Experiments with a New Boosting Algorithm?, 1996
  %     https://cseweb.ucsd.edu/~yfreund/papers/boostingexperiments.pdf
  % [2] AIMA book, Stuart Russell et.al. (second edition)
  % [3] Ji Zhu, Saharon Rosset, Hui Zou, Trevor Hastie, ?Multi-class AdaBoost?, January 12, 2006.
  %     https://web.stanford.edu/~hastie/Papers/samme.pdf
  properties (SetAccess = private, GetAccess = public)
    % weak_learner class with metchods:
    % obj = obj.train(x,y,w)
    % y = obj.predict(X)
    weak_learner % see above
  end
  properties (SetAccess = private, GetAccess = private)
    model            % array of trained weak_learners
    weights          % weight of each weak classifier
    labels           % original labels passed in the y array
    nFeat   = 1      % number of features in the train data set
    verbose = false; % print-out or not
  end % properties
  methods (Access = public)
    
    %% ============================================================
    function obj = AdaBoost_samme (weak_learner_class, verbose)
      obj.weak_learner = weak_learner_class;
      if nargin>1
        obj.verbose = verbose;
      end
    end
    
    %% ====================================================================
    function [obj, accuracy] = train(obj, X, y, w, nIter)
      % INPUT:
      % X - (N x D) train data set, each of N rows is a training
      %      sample in the D dimensional feature space.
      % y - (N x 1) true label corresponding to each sample. Only 2 unique
      %      labels are allowed.
      % w0- (N x 1) optional weights for each sample in case not all
      %      the points carry the same weight
      % nIter - is the number of weak_learners to use.
      % OUTPUT:
      % obj - trained model
      % accuracy - (# Iterations by # classes) each column
      % corresponds to of boolean decission accuracy per iteration
      % of each class against all the rest.
      
      %% In case of multi-class problem call several 2-class classifiers
      if isempty(w)
        w = ones(size(y)); % all samples have equal weight at the begining
      end
      [obj.labels, ~, y] = unique(y);
      nLabel = numel(obj.labels);
      obj.nFeat = size(X,2);
      %Commnent: This restriction is not necesary but simplifies
      %import/export methods
      assert(isnumeric(y), 'AdaBoost_Samme labels have to be numeric.');
      assert(size(X,1)==numel(y), 'Number of Rows in X has to be the same as number of elements in y.')
      
      %% Implement N-class classifier
      nSample = size(X,1);
      w = w/sum(w); % sample weights adds up to 1
      y = y(:);      % true labels
      obj.weak_learner = obj.weak_learner.preprocess_train_data(X);
      obj.weights = zeros(nIter,1);
      accuracy    = zeros(nIter,1);
      nPert = 10;
      prob = zeros(nSample,nLabel); % likelihood that sample is yl
      for iIter = 1:nIter
        % Fit "Weak Learner classifier" to the training data using with 
        % weight distribution w 
        obj.model{iIter,1} = obj.weak_learner.train(X,y,w);
        Y = obj.model{iIter,1}.predict(X);
        Y = Y(:); % predicted labels
        mismatch = (Y~=y); % mismatch between predicted labels and true labeled samples
        
        % Calculate error
        error_rate = sum(w(mismatch)); % sum of weights of mismatched samples
        error_rate = min(max(error_rate,eps),1-eps); % clip to [eps, 1-eps] range
        
        % The weight of the iIter-th weak classifier
        % "log(nLabel-1)" is the unique contribution of SAMME algorithm
        alpha = log((1-error_rate)/error_rate) + log(nLabel-1);
        obj.weights(iIter,1) = alpha;
        
        % Importance of the true classified samples is decreased for the next weak classifier
        w(mismatch) = w(mismatch)*exp(alpha);
        if (abs(alpha)<1e-10) % in the unlucky case when alpha is close to zero perturb the weights
          msk = (randi(2,size(alpha))==1);     % pick random 50% of samples
          w(msk) = (nPert*w(msk)+1)/(nPert+1); % remap weights from [0, 1] to [1/nPert, 1]
          fprintf('perturb alpha=%e\n',alpha);
        end
        w = w/sum(w); % Renormalize so sum(w)=1
        
        % Track overall performance (not part of the algorithnm)
        if obj.verbose
          for k = 1:nLabel
            prob(:,k) = prob(:,k) + alpha * (Y==k);
          end
          [~, Y1] = max(prob, [],2); % predicted label
          match1 = (Y1==y); % true labeled samples
%           accuracy(iIter,1) = nnz(match1)/nSample;
%           fprintf('%3i) %s: weight(%3i)=%5.3f; stump accuracy(%3i)=%6.1f%%; overall accuracy=%6.1f%% %i misclassified\n', ...
%             iIter, obj.model{iIter,1}.print(), iIter, alpha, ...
%             iIter, 100*nnz(~mismatch)/nSample, 100*accuracy(iIter,1), nSample-nnz(match1));
        end
       end
    end % end train
    
    %% ====================================================================
     function [accuracy, tpr, P] = calc_accuracy(obj, X, y, nIter)
      % INPUT:
      % X - (N x D) train data set, each of N rows is a training
      % sample in the D dimensional feature space.
      % y - (N x 1) true label corresponding to each sample
      % OUTPUT:
      % accuracy - (# Iterations by 1) decission accuracy per iteration.
      % tpr - (# Iterations by # classes) true positive rate for each
      %   class corresponds to fraction of samples from each class which
      %   were classified correctly
      % P - (N x # Iterations) for each iterationwhat is the
      % probability that sample belongs to each class
      if nargin<4 || isempty(nIter)
        nIter = length(obj.weights);
      end
      accuracy = zeros (nIter,1);
      nSample = size(X,1);
      nLabel = numel(obj.labels);
      [~,~, y] = unique(y(:));
      tpr = zeros(nIter,2);
      n = hist(y,1:nLabel);
      prob = zeros(nSample,nLabel); % likelihood that sample is yl
      P = zeros(nSample,nIter,nLabel); % likelihood that it is y2 per iteration
      w_sum = 0;
      for iIter = 1:nIter
        % Call Weak Learner providing it with weight distribution w
        Y = obj.model{iIter,1}.predict(X);
        alpha = obj.weights(iIter,1);
        w_sum = w_sum + alpha;
        
        for k = 1:nLabel
          prob(:,k) = prob(:,k) + alpha * (Y==k);
          P(:,iIter, k) = prob(:,k);
        end
        [~, Y] = max(prob, [], 2); % predicted label
                 
        match = (Y==y); % mark matches
        accuracy(iIter,1) = nnz(match)/nSample;
        for k = 1:nLabel
          tpr(iIter,k) = nnz(match(y==k))/n(k);
        end
      end
    end % end calc_accuracy
    
    
    %% ====================================================================
    function [y, prob] = predict(obj,X,nIter)
      % INPUT:
      % X - (N x D) trest data set, each of N rows is a testing
      % sample in the D dimensional feature space.
      % nIter - optional parameter in case you want to try fewer
      % iterations than trained on.
      % OUTPUT:
      % y - (N x 1) each entry is the predicted label
      if nargin<3 || isempty(nIter)
        nIter = size (obj .model,1);
      else
        nIter = min(nIter,size(obj.model,1));
      end
      nLabel  = numel(obj.labels);
      nSample = size(X,1);
      prob = zeros(nSample,nLabel); % likelihood that sample is yl
      w_sum = 0;
      % for each weak classifier, likelihoods of test samples are collected
      for iIter=1:nIter
        Y = obj.model{iIter,1}.predict(X);
        w = obj.weights(iIter,1);       
        for k = 1:nLabel
          prob(:,k) = prob(:,k) + w * (Y==k);
        end
        w_sum = w_sum + w;
      end
      prob = prob/w_sum;
      [~,id] = max(prob,[],2); % weighted sum of each likelihood
      y = obj.labels(id);
    end % predict
    
    %% ====================================================================
    function [hx, hy] = feature_hist(obj)
      % which features do we use
      % OUTPUT:
      %  hx - feature number: array of integers 1 to # features
      %  hy - for each feature calculate sum of weights
      hx = 1:obj.nFeat;
      hy = zeros(size(hx));
      nIter = length(obj.model);
      for k = 1:nIter
        [~, d] = obj.model{k}.export_model;
        d = abs(d);
        hy(d) =hy(d) + obj.weights(k);
      end
    end
    
    %% ====================================================================
    function nIter = get_number_iterations (obj)
      % Output:
      % - nIter - scalar - number of iterations or weak classifiers used
      nIter = length(obj.model);
    end
    
    %% ====================================================================
    function [strList, labels, header] = export_model(obj)
      % Output:
      % - strList - List of strings each defining a weak classifier
      % - labels  - list of labels
      % - header  - header matching strList
      nIter = length(obj.model);
      strList = cell(nIter,1);
      for k = 1:nIter
        [str, ~, header] = obj.model{k}.export_model();
        strList{k} = sprintf('%s,%e', str, obj.weights(k));
      end
      labels = obj.labels(:)'; % row vector
      header = [header, ', weights'];
    end
    
    %% ========================================================
    function obj = import_model(obj, strList, labels)
      % Input:
      % - strList - List of strings each defining a weak classifier
      % - labels  - list of labels
      nIter = length(strList);
      obj.labels  = labels(:); % column vector
      obj.weights = zeros(nIter,1);
      for k = 1:nIter
        if length(obj.model) <k
          obj.model{k,1} = obj.weak_learner;
        end
        str = strList{k};
        j = find(str==',' ,1, 'last');
        obj.weights(k,1) = str2double(str(j+1:end)); % weight
        obj.model  {k,1} = obj.model{k}.import_model(str(1:j-1));
      end
      obj.model(nIter+1:end) = [];
    end % import_model
  end % methods
end % classdef
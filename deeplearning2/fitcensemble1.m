function obj = fitcensemble1(X,Y,varargin)
%  fitcensemble Fit ensemble of classification learners
%     ENS=fitcensemble(TBL,Y) fits a classification ensemble model ENS,
%     which can be used for making predictions on new data. ENS uses a
%     collection of individual classification learners such as
%     classification trees. These individual learners are trained from one
%     or more templates specified in LEARNERS.
%
%     TBL is a table containing predictors and Y is the response. Y can be
%     any of the following:
%       1. An array of class labels. Y can be a categorical array, logical
%          vector, numeric vector, string array or cell array of strings.
%       2. The name of a variable in TBL. This variable is used as the
%          response Y, and the remaining variables in TBL are used as
%          predictors.
%       3. A formula string such as 'y ~ x1 + x2 + x3' specifying that the
%          variable y is to be used as the response, and the other variables
%          in the formula are predictors. Any table variables not listed in
%          the formula are not used.
%
%     ENS=fitcensemble(X,Y) is an alternative syntax that accepts X as an
%     N-by-P matrix of predictors with one row per observation and one
%     column per predictor. Y is the response.
%
%     Use of a matrix X rather than a table TBL saves both memory and
%     execution time.
%
%     ENS is an object of class ClassificationEnsemble. If you use one of
%     the following five options and do not pass OptimizeHyperparameters,
%     ENS is of class ClassificationPartitionedEnsemble: 'CrossVal',
%     'KFold', 'Holdout', 'Leaveout' or 'CVPartition'.
%
%     ENS=fitcensemble(...,'PARAM1',val1,'PARAM2',val2,...) specifies
%     optional parameter name/value pairs:
%       'Method'    - Learner aggregation method. Method must be one of
%                     the following, case-insensitive character vectors:
%                     For binary classification:
%                       'AdaBoostM1'
%                       'LogitBoost'
%                       'GentleBoost'
%                       'RobustBoost' (requires Optimization Toolbox)
%                       'LPBoost' (requires Optimization Toolbox)
%                       'RUSBoost'
%                       'TotalBoost' (requires Optimization Toolbox)
%                       'Bag'
%                       'Subspace'
%                     For classification with 3 or more classes:
%                       'AdaBoostM2'
%                       'LPBoost' (requires Optimization Toolbox)
%                       'RUSBoost'
%                       'TotalBoost' (requires Optimization Toolbox)
%                       'Bag'
%                       'Subspace'
%                     Default: 
%                       'Subspace' if 'Learners' are only KNNs and
%                       Discriminants; 'AdaBoostM1' or 'AdaBoostM2' if
%                       'Learners' are Discriminants and Trees;
%                       'Logitboost' or 'AdaboostM2' if 'Learners' are all
%                       Trees.
%       'NumLearningCycles'    - Positive integer or 'AllPredictorCombinations'
%                     specifying the number of ensemble learning cycles. At
%                     every training cycle, fitcensemble loops over all
%                     learner templates in Learners and trains one weak
%                     learner for every template object. For positive
%                     integers, the number of trained learners in ENS is
%                     equal to NumLearningCycles*numel(Learners). To set
%                     'AllPredictorCombinations', Method must be 'Subspace'
%                     and Learners must represent one learner. In this
%                     case, fitcensemble trains
%                     NCHOOSEK(size(X,2),NPredToSample) learners for all
%                     possible combinations of NPredToSample predictors.
%                     Usually, an ensemble with a good predictive power
%                     needs between a few hundred and a few thousand weak
%                     learners. You do not have to train an ensemble for
%                     that many cycles at once. You can start by growing
%                     a few dozen learners, inspect the ensemble
%                     performance and, if necessary, train more weak
%                     learners using RESUME method of the ensemble.
%                     Default: 100
%       'Learners'  - Character vector specifying the name of the weak
%                     learner, or a single or cell array of weak learner
%                     template objects. When specifying template objects,
%                     you must construct every one using the appropriate
%                     learner template function. For example, call
%                     TEMPLATETREE if you want to grow an ensemble of
%                     trees. Usually, you need to supply only one weak
%                     learner template. To supply one weak learner using
%                     default parameters, pass Learners in as a character
%                     vector specifying the name of the weak learner, for
%                     example, 'tree'.
%                     Use the following learner names and templates:
%                       'Discriminant'          templateDiscriminant
%                       'KNN'                   templateKNN
%                       'Tree'                  templateTree
%                     Note: Ensemble performance depends on the parameters
%                     of the weak learners. You can get poor performance
%                     for weak learners with default parameters. 
%                     Default: 
%                       'KNN' when Method is 'Subspace',
%                       'Tree' when Method is 'Bag', 
%                       templateTree('MaxNumSplits',10) when Method is a 
%                           boosting method.
%       'NPrint'    - Print-out frequency, a positive integer scalar.
%                     By default, this parameter is set to 'off' (no
%                     print-outs). You can use this parameter to keep track
%                     of how many weak learners have been trained, so far.
%                     This is useful when you train ensembles with many
%                     learners on large datasets.
%       'OptimizeHyperparameters' 
%                      - Hyperparameters to optimize. Either 'none',
%                        'auto', 'all', a string/cell array of eligible
%                        hyperparameter names, or a vector of
%                        optimizableVariable objects, such as that returned
%                        by the 'hyperparameters' function. To control
%                        other aspects of the optimization, use the
%                        HyperparameterOptimizationOptions name-value pair.
%                        'auto' is equivalent to {'LearnRate', 'Method',
%                        'NumLearningCycles'}, plus the 'auto'
%                        hyperparameters of the specified weak learner.
%                        'all' is equivalent to {'LearnRate', 'Method',
%                        'NumLearningCycles'}, plus all eligible
%                        hyperparameters of the specified weak learner.
%                        Default: 'none'.
%   Refer to the MATLAB documentation for info on parameters for
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleGeneralEnsembleOptions')">classification ensembles</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleClassificationOptions')">classification (such as 'Prior', 'Cost' and others)</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleCVOptions')">cross-validation</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleSampleOptions')">sampling options for boosting and bagging</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleBoostOptions')">AdaBoostM1, AdaBoostM2, LogitBoost, and GentleBoost</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleRUSBoostOptions')">RUSBoost</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleLPTotalBoostOptions')">LPBoost and TotalBoost</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleRobustBoostOptions')">RobustBoost</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleRandomSubspaceOptions')">Random subspace</a>
%       <a href="matlab:helpview(fullfile(docroot,'stats','stats.map'), 'fitcensembleHyperparameterOptimizationOptions')">Hyperparameter Optimization</a>
%
%   Example 1: Train and inspect resubstitution error
%      load ionosphere;
%      ada = fitcensemble(X,Y,'Method','AdaBoostM1','NumLearningCycles',100);
%      plot(resubLoss(ada,'mode','cumulative'));
%      xlabel('Number of decision trees');
%      ylabel('Resubstitution error');
%
%   Example 2: Train and estimate generalization error on a holdout sample
%      load ionosphere;
%      ada = fitcensemble(X,Y,'Method','AdaBoostM1','NumLearningCycles',100,'Holdout',0.5);
%      plot(kfoldLoss(ada,'mode','cumulative'));
%      xlabel('Number of decision trees');
%      ylabel('Holdout error');
%
%   See also templateTree, templateKNN, templateDiscriminant,
%   classreg.learning.classif.ClassificationEnsemble,
%   classreg.learning.classif.ClassificationEnsemble/resume,
%   classreg.learning.partition.ClassificationPartitionedEnsemble

%   Copyright 2017 The MathWorks, Inc.

checkNotTall1(upper(mfilename),0,X,Y,varargin{:});

if nargin > 1
    Y = convertStringsToChars(Y);
end

if nargin > 2
    [varargin{:}] = convertStringsToChars(varargin{:});
end

[IsOptimizing, RemainingArgs] = parseOptimizationArgs1(varargin);
if IsOptimizing
    obj = classreg.learning.paramoptim.fitoptimizing('fitcensemble1',X,Y,varargin{:});
else
    Names = {'Method', 'NumLearningCycles', 'Learners'};
    Defaults = {[], 100, []};
    [Method, NumLearningCycles, Learners, ~, RemainingArgs] = parseArgs1(...
        Names, Defaults, RemainingArgs{:});
    if ~isempty(Learners)
        checkLearners(Learners);
    end
    if isempty(Method)
        Method = chooseDefaultMethod(Learners, X, Y, RemainingArgs);
    else
        checkMethod(Method);
    end
    if isempty(Learners)
        Learners = chooseDefaultLearners(Method);
    end
    if isBoostingMethod(Method)
        Learners = setTreeDefaultsIfAny(Learners);
    end
    obj = fitensemble1(X, Y, Method, NumLearningCycles, Learners, ...
        RemainingArgs{:}, 'Type', 'classification');
end
end

function checkMethod(Method)
if ~ischar(Method)
    error(message('stats:fitensemble1:MethodNameNotChar'));
end
if ~any(strncmpi(Method,ensembleModels1(),length(Method)))
    error(message('stats:fitensemble1:BadMethod', Method));
end
end

function checkLearners(Learners)
if ~(ischar(Learners) || isa(Learners, 'classreg.learning.FitTemplate1') || ...
        iscell(Learners) && all(cellfun(@(Tmp)isa(Tmp, 'classreg.learning.FitTemplate1'), Learners)))
    %error(message('stats:fitensemble1:BadLearners'));
end
end

function Method = chooseDefaultMethod(Learners, X, Y, NVPs)
if onlyLearnerTypes(Learners, {'knn','discriminant'})
    Method = 'Subspace';
elseif allLearnerTypes(Learners, {'tree','discriminant'})
    if numClasses(X, Y, NVPs) > 2
        Method = 'AdaboostM2';
    else
        Method = 'AdaBoostM1';
    end
else
    % All trees
    if numClasses(X, Y, NVPs) > 2
        Method = 'AdaboostM2';
    else
        Method = 'LogitBoost';
    end
end
end

function tf = isBoostingMethod(Method)
tf = ischar(Method) && ~isempty(strfind(lower(Method), 'boost'));
end

function tf = onlyLearnerTypes(Learners, Types)
% Return true if the only learner types are in Types.
if isempty(Learners)
    tf = false;
elseif ischar(Learners)
    tf = ismember(lower(Learners), Types);
elseif isa(Learners, 'classreg.learning.FitTemplate1')
    tf = ismember(lower(Learners.Method), Types);
elseif iscell(Learners)
    tf = all(cellfun(@(Template)ismember(lower(Template.Method), Types), ...
                     Learners));
else
    tf = false;
end
end

function tf = allLearnerTypes(Learners, RequiredTypes)
% Return true if all learner types in RequiredTypes are present.
if isempty(Learners)
    tf = false;
elseif ischar(Learners)
    tf = all(ismember(RequiredTypes, lower(Learners)));
elseif isa(Learners, 'classreg.learning.FitTemplate1')
    tf = all(ismember(RequiredTypes, lower(Learners.Method)));
elseif iscell(Learners)
    tf = all(ismember(RequiredTypes, cellfun(@(Template)lower(Template.Method), Learners, 'UniformOutput', false)));
else
    tf = false;
end
end

function Learners = chooseDefaultLearners(Method)
if ischar(Method) && isequal(lower(Method), 'subspace')
    Learners = 'KNN';
else
    Learners = 'Tree';
end
end

function N = numClasses(X, Y, NVPs)
[ClassNamesPassed, ~, ~] = internal.stats.parseArgs({'ClassNames'}, {[]}, NVPs{:});
if isempty(ClassNamesPassed)
    [~,Y] = classreg.learning.internal.table2FitMatrix(X,Y,NVPs{:});
    N = numel(levels(classreg.learning.internal.ClassLabel(Y)));
else
    N = numel(levels(classreg.learning.internal.ClassLabel(ClassNamesPassed)));
end
end

function Learners = setTreeDefaultsIfAny(Learners)
% For any learners that are trees, make MaxNumSplits default to 10. 
if ischar(Learners) && isequal(lower(Learners), 'tree')
    Learners = templateTree('MaxNumSplits', 10);
elseif isa(Learners, 'classreg.learning.FitTemplate1') 
    Learners = defaultMaxNumSplitsIfTemplateTree(Learners, 10);
elseif iscell(Learners) && all(cellfun(@(Tmp)isa(Tmp, 'classreg.learning.FitTemplate1'), Learners))
    Learners = cellfun(@(Tmp)defaultMaxNumSplitsIfTemplateTree(Tmp, 10), ...
                       Learners, 'UniformOutput', false);
end
end

function Tmp = defaultMaxNumSplitsIfTemplateTree(Tmp, value)
if isequal(lower(Tmp.Method), 'tree')
    Tmp = fillIfNeeded(Tmp, 'classification');
    if isempty(getInputArg(Tmp, 'MaxSplits'))
        Tmp = setInputArg(Tmp, 'MaxSplits', value);
    end
end
end

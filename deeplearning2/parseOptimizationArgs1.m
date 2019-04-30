function [IsOptimizing, RemainingArgs] = parseOptimizationArgs1(Args)
    
%   Copyright 2016 The MathWorks, Inc.

[OptimizeHyperparameters,~,~,RemainingArgs] = parseArgs1(...
    {'OptimizeHyperparameters', 'HyperparameterOptimizationOptions'}, {[], []}, Args{:});
IsOptimizing = ~isempty(OptimizeHyperparameters) && ~isPrefixEqual(OptimizeHyperparameters, 'none');
end

function tf = isPrefixEqual(thing, targetString)
tf = ~isempty(thing) && ischar(thing) && strncmpi(thing, targetString, length(thing));
end
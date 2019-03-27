function tf = istall1(x)
%ISTALL Is the supplied array a tall array or tall table
%   OUT = ISTALL(X) returns TRUE if X is a tall array or a tall table, otherwise
%   it returns FALSE.
%
%   See also: tall.

% Copyright 2015 The MathWorks, Inc.

tf = isa(x, 'tall');
end

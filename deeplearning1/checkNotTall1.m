function checkNotTall1(fcn, ~, varargin)
%   checkNotTall Throw an error if any trailing argument is tall
%   checkNotTall(FCN,OFFSET,V1,V2,...) throws (as caller) error(message(ID,
%   FCN)) if any of V1,V2,... is tall. OFFSET is the number of arguments to the
%   original function prior to those input to this function.
%   The only difference to function tall.checkNotTall() is this one throws
%   out a different error message, and this one also ignore the second
%   input.

% Copyright 2016-2017 The MathWorks, Inc.

firstTallArg = find(cellfun(@istall1, varargin), 1, 'first');
if ~isempty(firstTallArg)
    msg = message('MATLAB:bigdata:array:FcnNotSupported', fcn);
    throwAsCaller(MException(msg.Identifier, '%s', getString(msg)));
end
end

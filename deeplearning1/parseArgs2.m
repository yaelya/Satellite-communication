function [varargout]=parseArgs2(pnames,dflts,varargin)
%parseArgs Process parameter name/value pairs for statistics functions
%   [A,B,...] = parseArgs(PNAMES,DFLTS,'NAME1',VAL1,'NAME2',VAL2,...)
%   In typical use there are N output values, where PNAMES is a cell array
%   of N valid parameter names, and DFLTS is a cell array of N default
%   values for these parameters. The remaining arguments are parameter
%   name/value pairs that were passed into the caller. The N outputs
%   [A,B,...] are assigned in the same order as the names in PNAMES.
%   Outputs corresponding to entries in PNAMES that are not specified
%   in the name/value pairs are set to the corresponding value from DFLTS.
%   Unrecognized name/value pairs are an error.
%
%   Each element of PNAMES is either a string with a parameter name or a
%   cell array of strings with possible names for the same parameter.
%
%   [A,B,...,SETFLAG] = parseArgs(...), where SETFLAG is the N+1 output
%   argument, also returns a structure with a field for each parameter
%   name. The value of the field indicates whether that parameter was
%   specified in the name/value pairs (true) or taken from the defaults
%   (false).
%
%   [A,B,...,SETFLAG,EXTRA] = parseArgs(...), where EXTRA is the N+2 output
%   argument, accepts parameter names that are not listed in PNAMES. These
%   are returned in the output EXTRA as a cell array.
%
%   Example:
%       pnames = {'color' 'linestyle', 'linewidth'}
%       dflts  = {    'r'         '_'          '1'}
%       varargin = {'linew' 2 'linestyle' ':'}
%       [c,ls,lw] = statslib.internal.parseArgs(pnames,dflts,varargin{:})
%       % On return, c='r', ls=':', lw=2
%
%       [c,ls,lw,sf] = statslib.internal.parseArgs(pnames,dflts,varargin{:})
%       % On return, sf = [false true true]
%
%       varargin = {'linew' 2 'linestyle' ':' 'special' 99}
%       [c,ls,lw,sf,ex] = statslib.internal.parseArgs(pnames,dflts,varargin{:})
%       % On return, ex = {'special' 99}

%   Copyright 2010-2014 The MathWorks, Inc.

% PNAMES can be (a) a char array, (b) a cellstr, or (c) a cell array with
% char arrays and/or cellstrs. This last case is for backwards
% compatibility for deprecated parameter/property names.
% Account for the unlikely case (a) first:
if ~iscell(pnames)
    pnames = {pnames};
end
% Unroll PNAMES in case of (c) and setup a backtrack index (bi)
nparams = length(pnames);
doBacktrack = false;
bi = ones(nparams,1);
for j=1:nparams
    if iscell(pnames{j})
        bi(j) = numel(pnames{j});
        doBacktrack = true;
    end
end
if doBacktrack
    bi = repelem(1:nparams,bi)';
    pnames = [pnames{:}];
end

% Initialize some other variables
varargout = dflts;
setflag = false(1,nparams);
unrecog = {};
nargs = length(varargin);

dosetflag = nargout>nparams;
dounrecog = nargout>(nparams+1);

% Must have name/value pairs
if mod(nargs,2)~=0
    m = message('stats:internal:parseArgs2:WrongNumberArgs');
    throwAsCaller(MException(m.Identifier, '%s', getString(m)));
end

% Process name/value pairs
for j=1:2:nargs
    pname = varargin{j};
    if ~ischar(pname)
        m = message('stats:internal:parseArgs2:IllegalParamName');
        throwAsCaller(MException(m.Identifier, '%s', getString(m)));
    end
    
    mask = strncmpi(pname,pnames,length(pname)); % look for partial match
    if doBacktrack
        mask = accumarray(bi,mask)>0;
    end
    if ~any(mask)
        if dounrecog
            % if they've asked to get back unrecognized names/values, add this
            % one to the list
            unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};
            continue
        else
            % otherwise, it's an error
            m = message('stats:internal:parseArgs2:BadParamName',pname);
            throwAsCaller(MException(m.Identifier, '%s', getString(m)));
        end
    elseif sum(mask)>1
        mask = strcmpi(pname,pnames); % use exact match to resolve ambiguity
        if doBacktrack
            mask = accumarray(bi,mask)>0;
        end
        if sum(mask)~=1
            m = message('stats:internal:parseArgs2:AmbiguousParamName',pname);
            throwAsCaller(MException(m.Identifier, '%s', getString(m)));
        end
    end
    varargout{mask} = varargin{j+1};
    setflag(mask) = true;
end

% Return extra stuff if requested
if dosetflag
    % If there are cells in PNAMES (case (c)), take the 1st string from
    % each cell.
    if doBacktrack
        pnames = pnames(diff([0;bi])>0);
    end
    setflag = cell2struct(num2cell1(setflag),pnames,2);
    
    varargout{nparams+1} = setflag;
    if dounrecog
        varargout{nparams+2} = unrecog;
    end
end
end


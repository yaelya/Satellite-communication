function [cm,gn] = confusionmat1(g,ghat,varargin)
% CONFUSIONMAT Confusion matrix for classification algorithms.
%    CM = CONFUSIONMAT(G,GHAT) returns the confusion matrix CM determined
%    by the known group labels G and the predicted group labels GHAT. G and
%    GHAT are grouping variables with the same number of observations. G
%    and GHAT can be categorical, numeric, or logical vectors;
%    single-column cell arrays of character vectors; single-column of
%    strings; or character matrices (each row representing a group label).
%    G and GHAT must be of the same type. CM is a square matrix with size
%    equal to the total number of distinct elements in G and GHAT. CM(I,J)
%    represents the count of instances whose known group labels are group I
%    and whose predicted group labels are group J. CONFUSIONMAT treats
%    NaNs, empty strings or 'undefined' values in G or GHAT as missing
%    values, and the corresponding observations are not counted. If inputs
%    are character matrices or cell string of charater arrays, CONFUSIONMAT
%    will trim inputs by strtrim().
%
%    The sets of groups and the orders of group labels in rows and
%    columns of CM are the same. They include all the groups appearing in
%    GN, and have the same order of group labels as GN, where GN is the
%    second of output of grp2idx([G;GHAT]).
%
%    CM = CONFUSIONMAT(G,GHAT,'ORDER',ORDER) returns the confusion matrix
%    with the order of rows (and columns) specified by ORDER.  ORDER is a
%    vector containing group labels and whose values can be compared to
%    those in G or GHAT using the equality operator. ORDER must contain all
%    the labels appearing in G or GHAT. ORDER can contain labels which do
%    not appear in G and GHAT, and hence CM will have zeros in the
%    corresponding rows and columns. If ORDER is a character matrix or cell
%    string of character array, CONFUSIONMAT will trim it by strtrim(). 
%
%    [CM, GORDER] = CONFUSIONMAT(G, GHAT) returns the order of group labels
%    for rows and columns of CM. GORDER has the same type as G and GHAT.
%
%   Example:
%      % Compute the resubstitution confusion matrix for applying CLASSIFY
%      % on Fisher iris data.
%      load fisheriris
%      x = meas;
%      y = species;
%      yhat = classify(x,x,y);
%      [cm,order] = confusionmat(y,yhat);
%
%   See also CROSSTAB, GRP2IDX.

%   Copyright 2008-2018 The MathWorks, Inc.


if nargin < 2
    iError(message('stats:confusionmat1:NotEnoughInputs'));
end

% Convert the Name 'Order' to char.
if nargin > 2
    [varargin{1}] = convertStringsToChars(varargin{1});
end

% Convert ghat to g's class if necessary
if ~isempty(g) && ~isempty(ghat) && ~strcmp(class(g),class(ghat))
    if iscellstr(g) && (isstring(ghat) || ischar(ghat)) && ismatrix(ghat)
        ghat = cellstr(ghat);
    elseif isstring(g) && (iscellstr(ghat) || ischar(ghat))
        ghat = string(ghat);
    elseif ischar(g) && (iscellstr(ghat) || isstring(ghat))
        ghat = char(ghat);
    end
end

gClass = class(g);
if ~strcmp(gClass,class(ghat))
    iError(message('stats:confusionmat1:GTypeMismatch'));
end

if ~isnumeric(g) && ~islogical(g) && ~isa(g,'categorical') ...
        && ~iscellstr(g)  && ~ischar(g) && ~isstring(g)
    iError(message('stats:confusionmat1:GTypeIncorrect'));
end

if ischar(g)
    if ~ismatrix(g) || ~ismatrix(ghat)
        iError(message('stats:confusionmat1:BadGroup'));
    end
    g = cellstr(g);
    ghat = cellstr(ghat);
elseif ~isvector(g) || ~isvector(ghat) 
    iError(message('stats:confusionmat1:BadGroup'));
else
    g = g(:);
    ghat = ghat(:);
end

if iscellstr(g)
    g = strtrim(g);
    ghat = strtrim(ghat);
end

if size(g,1) ~= size(ghat,1)
    iError(message('stats:confusionmat1:GRowNumMismatch'));
end

if isa(g,'categorical')
    if isordinal(g)
        if ~isordinal(ghat)
            iError(message('stats:confusionmat1:GOrdinalLevelsMismatch'));
        elseif ~isequal(categories(g),categories(ghat))
            iError(message('stats:confusionmat1:GOrdinalLevelsMismatch'));
        end
    elseif isordinal(ghat)
        iError(message('stats:confusionmat1:GOrdinalLevelsMismatch'));
    end
end


order = iParseOrderNameValuePair(varargin{:});


% Convert order to g's class if necessary
if ~isempty(order) && ~strcmp(class(g),class(order))
    if iscellstr(g) && (isstring(order) || ischar(order)) && ismatrix(order)
        order = cellstr(order);
    elseif isstring(g) && (iscellstr(order) || ischar(order))
        order = string(order);
    elseif ischar(g) && (isstring(order) || ischar(order))
        order = char(order);
    end
end

if ~isempty(order)
    if ischar(order)
        if ~ismatrix(order)
            iError(message('stats:confusionmat1:NDCharArrayORDER'));
        end
        order = cellstr(order);
    elseif ~isvector(order)
        iError(message('stats:confusionmat1:NonVectorORDER'));
    end

    if isa(g,'categorical')
        if iscellstr(order)
            if any(strcmp('',strtrim(order)))
                iError(message('stats:confusionmat1:OrderHasEmptyString'));
            end
        elseif iscategorical(order)
            if any(isundefined(order))
                iError(message('stats:confusionmat1:OrderHasUndefined'));
            end
        else
            iError(message('stats:confusionmat1:TypeMismatchOrder'));
        end
        
        g = setcatsLocal(g,order);
        ghat = setcatsLocal(ghat,order);
        
    else % g is not categorical vector

        if isnumeric(g)
            if ~isnumeric(order)
                iError(message('stats:confusionmat1:TypeMismatchOrder'));
            end
            if any(isnan(order))
                iError(message('stats:confusionmat1:OrderHasNaN'));
            end

        elseif islogical(g)
            if islogical(order)
                %OK. do nothing
            elseif isnumeric(order)
                if any(isnan(order))
                    iError(message('stats:confusionmat1:OrderHasNaN'));
                end

                order = logical(order);
                
            else
                iError(message('stats:confusionmat1:TypeMismatchOrder'));
            end

        elseif iscellstr(g)
            if ~iscellstr(order)
                iError(message('stats:confusionmat1:TypeMismatchLevels'));
            end
            if any(strcmp('',strtrim(order)))
                iError(message('stats:confusionmat1:OrderHasEmptyString'));
            end
            order = strtrim(order);
        end

        try
            uorder = unique(order);
        catch ME
            iError(message('stats:confusionmat1:UniqueMethodFailedOrder'));
        end

        if length(uorder) < length(order)
            iError(message('stats:confusionmat1:DuplicatedOrder'));
        end

        order = order(:);
    end
end

% Perform calculation
[cm, gn] = iCalculateConfusion(g, ghat);

if ~isempty(order)
    if ~isa(g,'categorical')
        %get the map from the default order to the given order
        [hasAllLabel,map] = ismember(gn,order);
        
        if ~all(hasAllLabel)
            iError(message('stats:confusionmat1:OrderInsufficientLabels'));
        end
        orderLen = length(order);
        cm2 = zeros(orderLen, orderLen);
        cm2(map,map) = cm(:,:);
        cm = cm2;

        if nargout > 1
            %convert gn to the same type as g
            if strcmp(gClass,'char')
                gn = char(order);
            else
                gn = order;
            end
        end
    end
elseif strcmp(gClass,'char')
    gn = char(gn);
end

end


function b = setcatsLocal(a,newCategories)
if iscategorical(newCategories)
    if ~isa(newCategories,class(a))
        iError(message('stats:confusionmat1:TypeMismatchOrder'));
    elseif isordinal(a)
        if ~isordinal(newCategories)
            iError(message('stats:confusionmat1:TypeMismatchOrder'));
        elseif ~isequal(categories(a),categories(newCategories))
            iError(message('stats:confusionmat1:TypeMismatchOrder'));
        end
    elseif isordinal(newCategories)
        iError(message('stats:confusionmat1:TypeMismatchOrder'));
    end
    newCategories = cellstr(newCategories);
end

existingCategories = categories(a);
if ~isempty(setdiff(existingCategories,newCategories))
    iError(message('stats:confusionmat1:OrderInsufficientLabels'));
end

b = addcats(a,newCategories,'after',existingCategories{end});
b = reordercats(b,newCategories);

end

function order = iParseOrderNameValuePair(varargin)
% Used for the confusionmat(g, ghat, 'Order', order) syntax. If no order
% was provided, returns empty. Otherwise, validates that the 'Order'
% parameter was provided.

switch nargin
    case 0
        order = [];
    case 2
        % Make sure the caller provided a string that fuzzy-matches 'Order'.
        validatestring(varargin{1}, {'Order'});
        
        % Assign the order.
        order = varargin{2};
    otherwise
        % Get the "WrongNumArgs" error message, throw it as if the error
        % came from confusionmat.m rather than this subfunction.
        throwAsCaller(iGetErrorWithWrongNumberOfArgs());
end

end

function [cm, classLabels] = iCalculateConfusion(g, ghat)
% Performs the confusion matrix calculation. g, ghat are the
% (validated) true and predicted labels.
% 
% Returns cm, the confusion matrix, and the labels of each class (the same
% type as g and ghat).

gLen = size(g,1);

% Use findgroups to obtain the group indexes, idx, and the group names,
% gLevels.
[idx,classLabels] = iFindGroups(g, ghat);

% Split the indices into true and predicted observations.
gidx = idx(1:gLen,:);
ghatidx = idx(gLen+1:gLen*2,:);

% Ignore NaN values in GIDX and GHATIDX
nanrows = isnan(gidx) | isnan(ghatidx);
if any(nanrows)
    gidx(nanrows,:) = [];
    ghatidx(nanrows,:) = [];
end

% Actual calculation of the confusion matrix.
cm = accumarray([gidx, ghatidx], 1, [length(classLabels), length(classLabels)]);

% If g is a cellstr or string array, we need to re-order gLevels and the
% confusion matrix so they are in order of classes first appearing (rather
% than alphabetical order, which is what findgroups returns).
if iscell(g) || isstring(g)
    [cm, classLabels] = reorderMatrixAndLabelsToFirstApperanceOrder(g, ghat, cm);
end

end

function [cm, classLabels] = reorderMatrixAndLabelsToFirstApperanceOrder(g, ghat, cm)
% If labels are cellstrings or string arrays, findgroups returns them in
% alphabetical order, but we want first appearance order. We need to
% reorder after calculating the matrix so we correctly deal with <missing>.

% Strip out any missing labels when we calculate the indices.
[levels,levelIdx] = unique(rmmissing([g;ghat]),'first');

% Sort by order of appearance
[~,idx] = sort(levelIdx);

% Re-order to order of appearance.
classLabels = levels(idx);
cm = cm(idx(:), idx(:));

end

function [idx,classLabels] = iFindGroups(g, ghat)
% Find the group indices idx, and the group names classLabels, using
% findgroups(). We want to make the following modifications:
% 
% 1: In the case where g and ghat are categoricals which define underlying
% categories that don't appear in the observations, we want classLabels to be
% the underlying categories (findgroups returns only categories seen in the
% observations); we deal with that here.
%
% 2: In the case where g and ghat are cellstrings, we want to return the
% categories in order of appearance (findgroups returns them in
% alphabetical order). However, this interacts with whether there are NaNs
% in the data - we deal with that after the confusion matrix
% has been calculated, and ignore it here.

[idx,classLabels] = findgroups([g;ghat]);

% If there are more underlying categories than are seen in the
% observations, get the list of underlying categories (in the order they
% appear).
if iscategorical(g) && length(categories([g; ghat])) > length(classLabels)
    % Converting to double is equivalent to finding the indices, including
    % underlying categories that aren't observed.
    idx = double([g; ghat]);
    
    % Get the categories (which will appear in the correct order) as a
    % categorical, preserving ordinality.
    classLabels = categorical(categories([g; ghat]), categories([g; ghat]), 'Ordinal', isordinal(g));
end

end

function iError(originalMsg)
% We want to throw an error with identifier "stats:confusionmat", even
% though the actual message catalog identifier is "mlearnlib:confusionmat".

oldID = string(originalMsg.Identifier);
newID = oldID.replace("stats", "mlearnlib");

% Get the error message from the new catalog but throw an error with the
% old ID.
newMsg = message(newID, originalMsg.Arguments{:});
errorText = newMsg.getString();
throwAsCaller(MException(oldID, errorText));
end

function mException = iGetErrorWithWrongNumberOfArgs()
% For the "WrongNumberArgs" error, we want to throw an error with ID 
% "stats:internal:parseArgs:WrongNumberArgs", even though the error message
% is actually located in "mlearnlib:confusionmat".

oldID = 'stats:internal:parseArgs:WrongNumberArgs';
newMsg = message('mlearnlib:confusionmat1:WrongNumberArgs');
errorText = newMsg.getString();

% We don't actually throw this error, but instead return the MException.
% This is because the error is being produced in a subfunction, and we want
% to throw it as caller.
mException = MException(oldID, errorText);
end
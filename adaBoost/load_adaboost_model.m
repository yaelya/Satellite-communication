function obj = load_adaboost_model(obj, filename)
fileID = fopen(filename, 'r');
%% read section 1
labels = [];
nIter  = zeros(1,1);
k = 0;
fgetl(fileID);          % read header
tline = fgetl(fileID);
while ischar(tline)
  k = k+1;
  if tline(1) == char('0'+k)
    [~,tline] = strtok(tline, ',');
    [v,tline] = strtok(tline(2:end), ',');
    nIter(k)  = str2double(v);
    L = str2num(['[', tline, ']']); %#ok<ST2NM>
    if isnumeric(L)
      labels(k,:) = L; % numeric labels
    else
      j = 0; % string labels
      while ~isempty(tline)
        j = j+1;
        [labels{k,j},tline] = strtok(tline, ','); %#ok<*STTOK>
      end
    end
  else
    break
  end
  tline = fgetl(fileID);
end
%% read section 2
strList = cell(sum(nIter),1);
for i = 1:sum(nIter)
  tline = fgetl(fileID);
  strList{i,1} = tline;
end
fclose(fileID);
obj = obj.import_model(strList, labels);
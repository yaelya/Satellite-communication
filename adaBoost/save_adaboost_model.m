function save_adaboost_model(obj, filename)
[strList, labels, header] = export_model(obj);
[nClassifier, nLabel] = size(labels);
nIter = size(strList,1)/nClassifier;
fileID = fopen(filename, 'w');
%% write section 1
hStr = 'Classifier#,nIter';
for i = 1:nLabel
  hStr = sprintf('%s,label #%i', hStr, i);
end
fprintf(fileID, '%s\n', hStr);
for iClass = 1:nClassifier
  lStr = sprintf('%i,%i', iClass, nIter);
  for i = 1:nLabel
    if isnumeric(labels)
      lStr = sprintf('%s,%i', lStr, labels(iClass,i));
    else
      lStr = sprintf('%s,%s', lStr, labels{iClass,i});
    end
  end
  fprintf(fileID, '%s\n', lStr);
end
%% write section 2
fprintf(fileID, '%s\n', header);
for i = 1:size(strList,1)
    fprintf(fileID, '%s\n', strList{i});
end
fclose(fileID);
function get_titanic_test(foldin, file, foldout, alreadydone)
% get_titanic_test('data', 'test', 'datamat', 0)

%% open the file and create folder for output, if needed
fid=fopen([foldin '/' file '.csv']);
[~]=fgetl(fid); 
if ~isdir(foldout)
    mkdir(foldout) 
end 

%% scan the file in large batch and do stuff
restofdata=[]; cp=0; tic
while ~feof(fid)
    bufferSize = 1e8; cp=cp+1;
    data = fread(fid,bufferSize,'uint8=>char')';
    data=[restofdata data];  %#ok<AGROW>
    [data, restofdata, list]=cleancut(data, cp);
    if cp>alreadydone
        dostuff(data, list, foldout, foldin, file, cp);
    end
    toc
end
fclose(fid);

end

function dostuff(data, list, foldout, foldin, file, cp)
% hard to be efficient
% do what you can 

lost=find(data(1:end-1)==','); 
lost=reshape(lost,size(lost,2)/size(list,2),size(list,2));

x(1:7,1:size(lost,2))=0;
y(1:size(lost,2))=0;

for i=1:size(lost,2)
    
    x(1,i)=i+891; % PassengerId 
    %y(i)=str2double(data(lost(1,i)+1:lost(2,i)-1)); % Survived (0 = No; 1 = Yes)
    x(2,i)=str2double(data(lost(1,i)+1:lost(2,i)-1)); % Pclass (1 = 1st; 2 = 2nd; 3 = 3rd)
    % Name => don't care
    x(3,i)=strcmp(data(lost(4,i)+1:lost(5,i)-1),'female'); % Sex
    x(4,i)=str2double(data(lost(5,i)+1:lost(6,i)-1)); % Age
    x(5,i)=str2double(data(lost(6,i)+1:lost(7,i)-1)); % SibSp Number of Siblings/Spouses Aboard
    x(6,i)=str2double(data(lost(7,i)+1:lost(8,i)-1)); % Parch Number of Parents/Children Aboard
    % Ticket => don't care
    x(7,i)=str2double(data(lost(9,i)+1:lost(10,i)-1)); % Passenger Fare
    % t11(i)= % Cabin => sure you don't care?
    switch data(lost(11,i)+1);
        % Port of Embarkation 
        % (C = Cherbourg; Q = Queenstown; S = Southampton)
        case 'C'
            x(8,i)=0;
        case 'Q'
            x(8,i)=1;
        case 'S'
            x(8,i)=2;
    end
    
end
    
save([foldout '/' foldin '_' file '_' int2str(cp) '.mat'], 'x', 'y');

end

function [data, restofdata, list]=cleancut(data, cp)
% segment the data and send back a clean cut and the remaining data
% assume no '*' character in the file
% and '\n' as the only wspace characters

if ~(cp-1); data=['*' data]; end
list=find(isstrprop(data,'cntrl'));
data(list(~(mod(1:size(list,2),(1:size(list,2))*0+2))))='*';
list=find(data=='*'); 
restofdata=data(list(end):end); 
data=data(1:list(end)-1); 
list=list(1:end-1);

end
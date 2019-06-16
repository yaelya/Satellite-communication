tic
clc; clear all
data=xlsread('34_35_1.xlsx');
Y = data(:,end);
X = data(:,1:end-1);

Z=data(randi([11,420],1,10))


for c= 1:size(Z)
    if Z(c)<0 || Z(c)>1
        flag=0
        break
    end
    flag=1
end

if flag==0
    for c= 1:11
        C2 = data(:,c) ;
        themin = min(C2) ;
        themax = max(C2) ;
        a(c,1)=themin;
        a(c,2)=themax;
        
    end
    xlswrite('myExample.xlsx',a,'MyData')
end

data2=xlsread('myExample.xlsx','MyData');
for c= 1:11
    if abs(data2(c,1)) > abs(data2(c,2))
        maximum= abs(data2(c,1));
        for d= 1:size(data)
            data(d,c)= abs(data(d,c))/ maximum;
        end
    end
    maximum= abs(data2(c,2));
    for d= 1:size(data)
        data(d,c)= abs(data(d,c))/ maximum;
    end         
end

file=xlsread('file1.xlsx');
file2=xlsread('file2.xlsx');


[y_DL]=DL_main(file2);
%SVM_main(file);
%adaboost_main(file);
%mkpred(file);
%pred_main(file);
toc
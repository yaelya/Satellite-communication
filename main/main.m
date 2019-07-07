tic
clc; clear all
data=xlsread('filet (1).xlsx');
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

file=xlsread('filet.xlsx');


[y_DL]=DL_main(file);
[y_SVM]=SVM_main(data);
[y_A]=adaboost_main(data);
[perftrain,perfage,perfage2,ley,y_RF]=mkpred(data);
%pred_main(file);

b(1,1)="DL";
b(1,2)="SVM";
b(1,3)="Adaboost";
b(1,4)="RF";
b(1,5)="SUM";

size=size(y_SVM);

for c= 1:size
    k=1;
    if y_DL(c)==0
        b(c+1,k)=-1;
    else
    b(c+1,k)=y_DL(c);
    end
end

for c= 1:size
    k=2;
    if y_SVM(c)==0
        b(c+1,k)=-1;
    else
    b(c+1,k)=y_SVM(c);
    end
end

for c= 1:size
    k=3;
    if y_A(c)==0
        b(c+1,k)=-1;
    else
    b(c+1,k)=y_A(c);
    end
end

for c= 1:size
    k=4;
    if y_RF(c)==0
        b(c+1,k)=-1;
    else
    b(c+1,k)=y_RF(c);
    end
end

xlswrite('resultes.xlsx',b,'MyData')

data3=xlsread('resultes.xlsx','MyData');

for c= 1:size
    k=1;
    if data3(c,k)==data3(c,k+1)
        b(c+1,5)=data3(c,k);
    elseif data3(c,k)==data3(c,k+2)
        b(c+1,5)=data3(c,k);
    elseif data3(c,k)==data3(c,k+3)
        b(c+1,5)=data3(c,k);
    else
        b(c+1,5)=data3(c,k+1);
    end
end

xlswrite('resultes.xlsx',b,'MyData');

toc
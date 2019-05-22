%https://www.mathworks.com/matlabcentral/fileexchange/63158-support-vector-machine
tic
clc; clear all
data=xlsread('file1.xlsx');

%disp(length(data));
data(:,1:end-1)=zscore(data(:,1:end-1));
[train,test] = holdout(data,80);
% Test set
Xtest=test(:,1:end-1);Ytest=test(:,end);
% Traing set
X=train(:,1:end-1);Y=train(:,end);
figure;
hold on;
scatter(X(Y==1,1),X(Y==1,2),'+g');
scatter(X(Y==-1,1),X(Y==-1,2),'.r');
xlabel('{x_1}');
ylabel('{x_2}');
legend('Positive Class','Negative Class');
title('Data for classification');
hold off;

fm_=[];
for c=[0.1]
     
     % alpha
     alpha = grad_ascent(X,Y,c);
     
     % Possible support vectors
     Xs=X(alpha>0,:); Ys=Y(alpha>0);
     
     % weights
     W=(alpha(alpha>0).*Ys)'*Xs;
     
     % bias
     bias=mean(Ys-(Xs*W'));
     
     % f~ (Predicted labels)
     f=sign(Xtest*W'+bias);
     
     % confusion matrix
     fm= confusion_mat(Ytest,f);
     fm_=[fm_; c fm];    
end

[max_fm, indx]=max(fm_(:,2));
c_optimal=fm_(indx,1);

alpha = grad_ascent(X,Y,c_optimal);
Xs=X(alpha>0,:); Ys=Y(alpha>0);
Support_vectors=size(Xs,1);
W=(alpha(alpha>0).*Ys)'*Xs;
bias=mean(Ys-(Xs*W'));    
f=sign(Xtest*W'+bias);
[F_measure, Accuracy] = confusion_mat(Ytest,f)
ft=X*W'+bias;
zeta=max(0,1-Y.*ft);
Non_Zero_Zeta=sum(zeta~=0);
Support_vectors;
figure;
hold on;
scatter(X(Y==1,1),X(Y==1,2),'b');
scatter(X(Y==-1,1),X(Y==-1,2),'r');
scatter(Xs(Ys==1,1),Xs(Ys==1,2),'.b');
scatter(Xs(Ys==-1,1),Xs(Ys==-1,2),'.r');
syms x;
fn=vpa((-bias-W(1)*x)/W(2),4);
fplot(fn,'Linewidth',2);
fn1=vpa((1-bias-W(1)*x)/W(2),4);
fplot(fn1,'--');
fn2=vpa((-1-bias-W(1)*x)/W(2),4);
fplot(fn2,'--');
axis([-2 2 -2 2]);
xlabel('X_1');
ylabel('X_2');
title('Hyperplane in 2D');
legend('+ve class','-ve class','support vector (+)','support vector (-)','Decision Boundry','Location','southeast');
hold off;
toc
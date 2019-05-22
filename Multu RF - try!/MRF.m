tic

total_train_time=0;
total_test_time=0;
 
data=xlsread('34_35.xlsx');
Y = data(:,end);
X = data(:,1:end-1);
rng(10,'twister');         % For reproducibility
[train,test] = holdout(data,80);

%twonorm, N=300, D=2
for i=1:10
	fprintf('%d,',i);
	tic;
	model=classRF_train(X,Y,8,1000,train(:,end),train(:,1:end-1));
    total_train_time=toc;
    tic;
	y_hat = classRF_predict(X,model);
% 	total_test_time=total_test_time+toc;	
%     length(find(y_hat~=outputs))/length(outputs)
%     keyboard
end
% fprintf('\nnum_tree %d: Avg train time %d, test time %d\n',1000,total_train_time/100,total_test_time/100);

toc
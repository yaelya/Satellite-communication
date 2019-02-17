function [ada_train, ada_test]= adaboost(Xtrain,Ytrain, Xtest)
% AdaBoost function 
% (X_train-> input: training set)
% (Y_train-> target)
% (Xtest-> input: testing set)
% (ada_train-> label: training set)
% (ada_test-> label: testing set)
% Choosen Weak classifiers:
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. Logistic Regression
% 5. SVM (rbf)
N=size(Xtrain,1);
a=[Xtrain Ytrain];
D=(1/N)*ones(N,1);
Dt=[]; h_=[];
Classifiers=5;
eps=zeros(Classifiers,1);
for T=1:Classifiers
    p_min=min(D);
    p_max=max(D);
    
    for i=1:length(D)
        p = (p_max-p_min)*rand(1) + p_min;
        
        if D(i)>=p
            d(i,:)=a(i,:);
        end
        
        t=randi(size(d,1));
        Dt=[Dt ;d(t,:)];
    end
    X=Dt(:,1:end-1);
    Y=Dt(:,end);
    
    if T==1
        % gda
        gda_in=fitcdiscr(X,Y);
        gda_out=predict(gda_in, X);
        h=gda_out;
        Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==2
        % knn with (30 Nearest Neighbour)
        knn_in=fitcknn(X,Y,'NumNeighbors',30);
        knn_out=predict(knn_in, X);
        h=knn_out;
        Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==3
        % nb
        nb_in=fitcnb(X,Y);
        nb_out=predict(nb_in, X);
        h=nb_out;
        Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==4
        % logistic regression
        linear_in=fitclinear(X,Y,'Learner','logistic');
        linear_out=predict(linear_in, X);
        h=linear_out;
        Dt=Dt(length(Dt)+1:end,:);
    end
    
    if T==5
        % svm 'rbf'
        svm_in=fitcsvm(X,Y,'KernelFunction','rbf');
        svm_out=predict(svm_in, X);
        h=svm_out;
        Dt=Dt(length(Dt)+1:end,:);
    end  
    
    h_=[h_ h];
    % weighted error
    for i=1:length(Y)
        if (h_(i,T)~=Y(i))
            eps(T)=eps(T)+D(i,:); 
        end  
    end
    
    % Hypothesis weight
    alpha(T)=0.5*log((1-eps(T))/eps(T));
    
    % Update weights
    D=D.*exp((-1).*Y.*alpha(T).*h);
    D=D./sum(D);
end
% final vote
H(:,1)=predict(gda_in, Xtrain);
H(:,2)=predict(knn_in, Xtrain);
H(:,3)=predict(nb_in, Xtrain);
H(:,4)=predict(linear_in, Xtrain);
H(:,5)=predict(svm_in, Xtrain);
ada_train(:,1)=sign(H*alpha');
% for test set
Htest(:,1)=predict(gda_in, Xtest);
Htest(:,2)=predict(knn_in, Xtest);
Htest(:,3)=predict(nb_in, Xtest);
Htest(:,4)=predict(linear_in, Xtest);
Htest(:,5)=predict(svm_in, Xtest);
ada_test(:,1)=sign(Htest*alpha');
end

%function pred_main(data)
    function [perftrain,perfage,perfage2,ley]=mkpred
        
        tic
        clc; clear all
        %data_matrix=xlsread('file1.xlsx');
        %icol = size(data_matrix,2);
        data_matrix=xlsread('file1.xlsx');
        icol = size(data_matrix,2)
        
        data_predictor = data_matrix(:,1:icol-1); % predictors matrix
        label = data_matrix(:,end); % last column is 2 for benign, 4 for malignant
        
        
        cvp=cvpartition(length(data_predictor),'holdout',0.2);
        %Training set
        x=data_predictor(training(cvp),:);
        y=label(training(cvp),:);
        
        %Testing set
        lex=data_predictor(test(cvp),:);
        ley=label(test(cvp),:);
        
        
        %load datamat/data_test_1.mat
        %lex=x';
        %ley=zeros(size(lex,1),1);
        
        %load datamat/data_train_1.mat
        %list=randperm(size(x,2));
        %x=x(:,list)';
        %y=y(list)';
        
        [lex, x, perfage,perfage2]=cleanage(lex,x);
        
        list=1:size(x,1); k=10; taille=floor(size(x,1)/k); perftrain(1:k)=0;
        for i=1:k
            listgen(1:taille)=list((i-1)*taille+1:i*taille);
            listtrain=setdiff(list,listgen);
            X_trn=x(listtrain,:);
            Y_trn=y(listtrain);
            X_tst=x(listgen,:);
            Y_tst=y(listgen);
            model = classRF_train(X_trn,Y_trn);
            ley = ley + classRF_predict(lex,model);
            Y_hat = classRF_predict(X_tst,model);
            perftrain(i)=length(find(Y_hat==Y_tst))/length(Y_tst);
            toc
        end
        
        ley=round(ley/k);
        
    end
    function [lex,x,perfage,perfage2]=cleanage(lex,x)
        
        letarget_x=[lex(isnan(lex(:,4)),2:3) lex(isnan(lex(:,4)),5:8)];
        letarget_y=zeros(size(letarget_x,1),1);
        
        target_x=[x(isnan(x(:,4)),2:3) x(isnan(x(:,4)),5:8)];
        target_y=zeros(size(target_x,1),1);
        
        y1=x(~isnan(x(:,4)),4);
        x1=[x(~isnan(x(:,4)),2:3) x(~isnan(x(:,4)),5:8)];
        
        list=1:size(x1,1); k=10; taille=floor(size(x1,1)/k);
        perfage(1:k)=0; perfage2(1:k)=0;
        for i=1:k
            
            listgen(1:taille)=list((i-1)*taille+1:i*taille);
            listtrain=setdiff(list,listgen);
            
            X_trn=x1(listtrain,:);
            Y_trn=y1(listtrain);
            X_tst=x1(listgen,:);
            Y_tst=y1(listgen);
            
            model = regRF_train(X_trn,Y_trn);
            Y_hat = regRF_predict(X_tst,model);
            
            perfage(i)=sqrt(sum((Y_hat-Y_tst).^2)/length(Y_tst));
            perfage2(i)=sqrt(sum((29.6991-Y_tst).^2)/length(Y_tst));
            
            target_y=target_y+regRF_predict(target_x,model);
            letarget_y=letarget_y+regRF_predict(letarget_x,model);
            
        end
        
        target_y=target_y/k;
        letarget_y=letarget_y/k;
        
        x(isnan(x(:,4)),4)=target_y;
        lex(isnan(lex(:,4)),4)=letarget_y;
        x=x(:,2:8); lex=lex(:,2:8);
        
    end
%end
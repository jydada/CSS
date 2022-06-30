clc;
clear all;
clear;
%% K锟斤拷锟节凤拷锟斤拷锟斤拷 锟斤拷KNN锟斤拷
load('feats_train.mat')
feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
cont=0;
for k=1:100
    [M_0,N_0]=size(feats_0);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_0=crossvalind('Kfold',feats_0(1:M_0,N_0),5);%进行随机分包
    test_0 = (indices_0 == 1); 
    train_0 = ~test_0;
    train_data_0=feats_0(train_0,:);
    train_target_0=iu_I_IIIlabel(train_0,:);
    test_data_0=feats_0(test_0,:);
    test_target_0=iu_I_IIIlabel(test_0,:);
    
    [M_1,N_1]=size(feats_1);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_1=crossvalind('Kfold',feats_1(1:M_1,N_1),5);%进行随机分包
    test_1 = (indices_1 == 1); 
    train_1 = ~test_1;
    train_data_1=feats_1(train_1,:);
    train_target_1=iu_IV_Vlabel(train_1,:);
    test_data_1=feats_1(test_1,:);
    test_target_1=iu_IV_Vlabel(test_1,:);
    
    train_data=[train_data_0;train_data_1];
    train_target=[train_target_0;train_target_1];
    
    test_data=[test_data_0;test_data_1];
    test_target=[test_target_0;test_target_1];
    
    model = ClassificationKNN.fit(train_data,train_target,'NumNeighbors',5);
    [Y,scores,~]= predict(model,test_data);
    for j=1:size(Y)
        predict_label(j+cont,1)=Y(j,1);
        decision_values(j+cont,1)=scores(j,2);
        a(j+cont,1)=(isequal(predict_label(j+cont,1),test_target(j,1)));
        test_label(j+cont,1)=test_target(j,1);
    end
    cont=size(predict_label,1);
end

TP=0;TN=0;FP=0;FN=0;
for i=1:size(predict_label,1)
    if isequal(predict_label(i,1),test_label(i,1))==1
        if isequal(predict_label{i,1},'pos')
            TP=TP+1;
        else
            TN=TN+1;
        end
    elseif isequal(predict_label{i,1},'pos')
            FP=FP+1;
        else
            FN=FN+1;
    end
end

[X,Y,T,AUC_KNN] = perfcurve(test_label,decision_values,'pos');
plot(X,Y,'r','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_KNN=sum(a(:))/size(a,1);
sensitivity_KNN=TP/(TP+FN);
specificity_KNN=TN/(TN+FP);
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
clear train_data;clear train_target;clear test_data;clear test_target;clear cont;
%% 锟斤拷锟缴拷址锟斤拷锟斤拷锟斤拷锟絉andom Forest锟斤拷
% load('feats_train.mat')
% feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
% feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
% cont=0;
% for k=1:100
%     [M_0,N_0]=size(feats_0);%数据集为一个M*N的矩阵，其中每一行代表一个样本
%     indices_0=crossvalind('Kfold',feats_0(1:M_0,N_0),5);%进行随机分包
%     test_0 = (indices_0 == 1); 
%     train_0 = ~test_0;
%     train_data_0=feats_0(train_0,:);
%     train_target_0=iu_I_IIIlabel(train_0,:);
%     test_data_0=feats_0(test_0,:);
%     test_target_0=iu_I_IIIlabel(test_0,:);
%     
%     [M_1,N_1]=size(feats_1);%数据集为一个M*N的矩阵，其中每一行代表一个样本
%     indices_1=crossvalind('Kfold',feats_1(1:M_1,N_1),5);%进行随机分包
%     test_1 = (indices_1 == 1); 
%     train_1 = ~test_1;
%     train_data_1=feats_1(train_1,:);
%     train_target_1=iu_IV_Vlabel(train_1,:);
%     test_data_1=feats_1(test_1,:);
%     test_target_1=iu_IV_Vlabel(test_1,:);
%     
%     train_data=[train_data_0;train_data_1];
%     train_target=[train_target_0;train_target_1];
%     
%     test_data=[test_data_0;test_data_1];
%     test_target=[test_target_0;test_target_1];
%     
%     model = TreeBagger(100,train_data,train_target);
%     [Y,scores,~]= predict(model,test_data);
%     for j=1:size(Y)
%         predict_label(j+cont,1)=Y(j,1);
%         decision_values(j+cont,1)=scores(j,2);
%         a(j+cont,1)=(isequal(predict_label(j+cont,1),test_target(j,1)));
%         test_label(j+cont,1)=test_target(j,1);
%     end
%     cont=size(predict_label,1);
% end
% TP=0;TN=0;FP=0;FN=0;
% for i=1:size(predict_label,1)
%     if isequal(predict_label(i,1),test_label(i,1))==1
%         if isequal(predict_label{i,1},'pos')
%             TP=TP+1;
%         else
%             TN=TN+1;
%         end
%     elseif isequal(predict_label{i,1},'pos')
%             FP=FP+1;
%         else
%             FN=FN+1;
%     end
% end
% [X,Y,T,AUC_RF] = perfcurve(test_label,decision_values,'pos');
% plot(X,Y,'g','linewidth',1)
% xlabel('False positive rate'); ylabel('True positive rate')
% title('ROC for classification by different classifers')
% accuracy_RF=sum(a(:))/size(a,1);
% sensitivity_RF=TP/(TP+FN);
% specificity_RF=TN/(TN+FP);
% hold on;
% clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
% clear train_data;clear train_target;clear test_data;clear test_target;clear cont;
%% LDA锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟絛iscriminant analysis classifier锟斤拷
load('feats_train.mat')
feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
cont=0;
for k=1:100
    [M_0,N_0]=size(feats_0);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_0=crossvalind('Kfold',feats_0(1:M_0,N_0),5);%进行随机分包
    test_0 = (indices_0 == 1); 
    train_0 = ~test_0;
    train_data_0=feats_0(train_0,:);
    train_target_0=iu_I_IIIlabel(train_0,:);
    test_data_0=feats_0(test_0,:);
    test_target_0=iu_I_IIIlabel(test_0,:);
    
    [M_1,N_1]=size(feats_1);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_1=crossvalind('Kfold',feats_1(1:M_1,N_1),5);%进行随机分包
    test_1 = (indices_1 == 1); 
    train_1 = ~test_1;
    train_data_1=feats_1(train_1,:);
    train_target_1=iu_IV_Vlabel(train_1,:);
    test_data_1=feats_1(test_1,:);
    test_target_1=iu_IV_Vlabel(test_1,:);
    
    train_data=[train_data_0;train_data_1];
    train_target=[train_target_0;train_target_1];
    
    test_data=[test_data_0;test_data_1];
    test_target=[test_target_0;test_target_1];
    
    model = ClassificationDiscriminant.fit(train_data, train_target,'DiscrimType','diaglinear');
    [Y,scores,~]= predict(model,test_data);
    for j=1:size(Y)
        predict_label(j+cont,1)=Y(j,1);
        decision_values(j+cont,1)=scores(j,2);
        a(j+cont,1)=(isequal(predict_label(j+cont,1),test_target(j,1)));
        test_label(j+cont,1)=test_target(j,1);
    end
    cont=size(predict_label,1);
end
TP=0;TN=0;FP=0;FN=0;
for i=1:size(predict_label,1)
    if isequal(predict_label(i,1),test_label(i,1))==1
        if isequal(predict_label{i,1},'pos')
            TP=TP+1;
        else
            TN=TN+1;
        end
    elseif isequal(predict_label{i,1},'pos')
            FP=FP+1;
        else
            FN=FN+1;
    end
end
[X,Y,T,AUC_LDA] = perfcurve(test_label,decision_values,'pos');
plot(X,Y,'b','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_LDA=sum(a(:))/size(a,1);
sensitivity_LDA=TP/(TP+FN);
specificity_LDA=TN/(TN+FP);
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
clear train_data;clear train_target;clear test_data;clear test_target;clear cont;
%% QDA锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟絛iscriminant analysis classifier锟斤拷
load('feats_train.mat')
feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
cont=0;
for k=1:100
    [M_0,N_0]=size(feats_0);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_0=crossvalind('Kfold',feats_0(1:M_0,N_0),5);%进行随机分包
    test_0 = (indices_0 == 1); 
    train_0 = ~test_0;
    train_data_0=feats_0(train_0,:);
    train_target_0=iu_I_IIIlabel(train_0,:);
    test_data_0=feats_0(test_0,:);
    test_target_0=iu_I_IIIlabel(test_0,:);
    
    [M_1,N_1]=size(feats_1);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_1=crossvalind('Kfold',feats_1(1:M_1,N_1),5);%进行随机分包
    test_1 = (indices_1 == 1); 
    train_1 = ~test_1;
    train_data_1=feats_1(train_1,:);
    train_target_1=iu_IV_Vlabel(train_1,:);
    test_data_1=feats_1(test_1,:);
    test_target_1=iu_IV_Vlabel(test_1,:);
    
    train_data=[train_data_0;train_data_1];
    train_target=[train_target_0;train_target_1];
    
    test_data=[test_data_0;test_data_1];
    test_target=[test_target_0;test_target_1];
    
    model = ClassificationDiscriminant.fit(train_data, train_target,'DiscrimType','diagquadratic');
    [Y,scores,~]= predict(model,test_data);
    for j=1:size(Y)
        predict_label(j+cont,1)=Y(j,1);
        decision_values(j+cont,1)=scores(j,2);
        a(j+cont,1)=(isequal(predict_label(j+cont,1),test_target(j,1)));
        test_label(j+cont,1)=test_target(j,1);
    end
    cont=size(predict_label,1);
end
TP=0;TN=0;FP=0;FN=0;
for i=1:size(predict_label,1)
    if isequal(predict_label(i,1),test_label(i,1))==1
        if isequal(predict_label{i,1},'pos')
            TP=TP+1;
        else
            TN=TN+1;
        end
    elseif isequal(predict_label{i,1},'pos')
            FP=FP+1;
        else
            FN=FN+1;
    end
end
[X,Y,T,	AUC_QDA] = perfcurve(test_label,decision_values,'pos');
plot(X,Y,'m','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_QDA=sum(a(:))/size(a,1);
sensitivity_QDA=TP/(TP+FN);
specificity_QDA=TN/(TN+FP);
% auc_mrmr=plot_roc_V2(decision_values,label,'g');
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
clear train_data;clear train_target;clear test_data;clear test_target;clear cont;
%% SVM
load('feats_train.mat')
feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
cont=0;
for k=1:100
    [M_0,N_0]=size(feats_0);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_0=crossvalind('Kfold',feats_0(1:M_0,N_0),5);%进行随机分包
    test_0 = (indices_0 == 1); 
    train_0 = ~test_0;
    train_data_0=feats_0(train_0,:);
    train_target_0=iu_I_IIIlabel(train_0,:);
    test_data_0=feats_0(test_0,:);
    test_target_0=iu_I_IIIlabel(test_0,:);
    
    [M_1,N_1]=size(feats_1);%数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices_1=crossvalind('Kfold',feats_1(1:M_1,N_1),5);%进行随机分包
    test_1 = (indices_1 == 1); 
    train_1 = ~test_1;
    train_data_1=feats_1(train_1,:);
    train_target_1=iu_IV_Vlabel(train_1,:);
    test_data_1=feats_1(test_1,:);
    test_target_1=iu_IV_Vlabel(test_1,:);
    
    train_data=[train_data_0;train_data_1];
    train_target=[train_target_0;train_target_1];
    
    test_data=[test_data_0;test_data_1];
    test_target=[test_target_0;test_target_1];
    
    SVMModel  = fitcsvm(train_data,train_target,'ClassNames',{'neg','pos'});
    %SVMModel =  fitcsvm(X,Y,'ClassNames',{'negClass','posClass'},'Standardize',true,...
    %    'KernelFunction','rbf','BoxConstraint',1);
    [Y,scores,~]= predict(SVMModel,test_data);
    for j=1:size(Y)
        predict_label(j+cont,1)=Y(j,1);
        decision_values(j+cont,1)=scores(j,2);
        a(j+cont,1)=(isequal(predict_label(j+cont,1),test_target(j,1)));
        test_label(j+cont,1)=test_target(j,1);
    end
    cont=size(predict_label,1);
end
TP=0;TN=0;FP=0;FN=0;
for i=1:size(predict_label,1)
    if isequal(predict_label(i,1),test_label(i,1))==1
        if isequal(predict_label{i,1},'pos')
            TP=TP+1;
        else
            TN=TN+1;
        end
    elseif isequal(predict_label{i,1},'pos')
            FP=FP+1;
        else
            FN=FN+1;
    end
end
[X,Y,T,	AUC_SVM] = perfcurve(test_label,decision_values,'pos');
plot(X,Y,'g','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_SVM=sum(a(:))/size(a,1);
sensitivity_SVM=TP/(TP+FN);
specificity_SVM=TN/(TN+FP);
% auc_mrmr=plot_roc_V2(decision_values,label,'g');
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
clear train_data;clear train_target;clear test_data;clear test_target;clear cont;
%% plot
hLe = legend({['KNN (AUC=' num2str(AUC_KNN) ')'],['LDA (AUC=' num2str(AUC_LDA)  ')'],['QDA (AUC=' num2str(AUC_QDA)  ')'],['SVM (AUC='  num2str(AUC_SVM)  ')']},...
    'location', 'southeast');
hLe.FontSize = 10;
saveas(gcf,'./RocTrain.tif');
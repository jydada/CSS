clc;
clear all;
clear;
%% data preparation
load('feats_train.mat')
train_feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
train_feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
cont=0;

train_data=[train_feats_0;train_feats_1];
train_target=[iu_I_IIIlabel;iu_IV_Vlabel];

clear iu_I_IIIfeats; clear iu_IV_Vfeats;clear iu_I_IIIlabel;clear iu_IV_Vlabel;
load('feats_test.mat')

test_feats_0=iu_I_IIIfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);
test_feats_1=iu_IV_Vfeats(:,[1,6,7,11,12,14,16,19,21,26,29,32,36,37,41,42,43,44,46,47,51,54,56,59,66,71,81,82,86,87,92,93,94,96,97,101,102,106,107,111,112,121,126,127,131,134,136,139,143,146,147,148,149,150,152,153,155,157,158,160,162,163,165,168,170,175,177,178,180,185,188,190,191,193,195]);

test_data=[test_feats_0;test_feats_1];
test_target=[iu_I_IIIlabel;iu_IV_Vlabel];

clear iu_I_IIIfeats; clear iu_IV_Vfeats;clear iu_I_IIIlabel;clear iu_IV_Vlabel;
%% KNN
 model = ClassificationKNN.fit(train_data,train_target,'NumNeighbors',5);
[predict_label,decision_values,~]= predict(model,test_data);
 for j=1:size(predict_label)
        a(j,1)=(isequal(predict_label(j,1),test_target(j,1)));
        test_label(j,1)=test_target(j,1);
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

[X,Y,T,AUC_KNN] = perfcurve(test_label,decision_values(:,2),'pos');
plot(X,Y,'r','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_KNN=sum(a(:))/size(a,1);
sensitivity_KNN=TP/(TP+FN);
specificity_KNN=TN/(TN+FP);
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
% %% RF
%  model = TreeBagger(100,train_data,train_target);
% [predict_label,decision_values,~]= predict(model,test_data);
%  for j=1:size(predict_label)
%         a(j,1)=(isequal(predict_label(j,1),test_target(j,1)));
%         test_label(j,1)=test_target(j,1);
%  end
%  TP=0;TN=0;FP=0;FN=0;
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
% 
% [X,Y,T,AUC_RF] = perfcurve(test_label,decision_values(:,2),'pos');
% plot(X,Y,'g','linewidth',1)
% xlabel('False positive rate'); ylabel('True positive rate')
% title('ROC for classification by different classifers')
% accuracy_RF=sum(a(:))/size(a,1);
% sensitivity_RF=TP/(TP+FN);
% specificity_RF=TN/(TN+FP);
% hold on;
% clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
 %% LDA
    model = ClassificationDiscriminant.fit(train_data, train_target,'DiscrimType','diaglinear');
[predict_label,decision_values,~]= predict(model,test_data);
 for j=1:size(predict_label)
        a(j,1)=(isequal(predict_label(j,1),test_target(j,1)));
        test_label(j,1)=test_target(j,1);
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

[X,Y,T,AUC_LDA] = perfcurve(test_label,decision_values(:,2),'pos');
plot(X,Y,'b','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_LDA=sum(a(:))/size(a,1);
sensitivity_LDA=TP/(TP+FN);
specificity_LDA=TN/(TN+FP);
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
%% QDA
    model = ClassificationDiscriminant.fit(train_data, train_target,'DiscrimType','diagquadratic');
[predict_label,decision_values,~]= predict(model,test_data);
 for j=1:size(predict_label)
        a(j,1)=(isequal(predict_label(j,1),test_target(j,1)));
        test_label(j,1)=test_target(j,1);
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

[X,Y,T,AUC_QDA] = perfcurve(test_label,decision_values(:,2),'pos');
plot(X,Y,'m','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_QDA=sum(a(:))/size(a,1);
sensitivity_QDA=TP/(TP+FN);
specificity_QDA=TN/(TN+FP);
% auc_mrmr=plot_roc_V2(decision_values,label,'g');
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
%% SVM
    SVMModel  = fitcsvm(train_data,train_target,'ClassNames',{'neg','pos'});
[predict_label,decision_values,~]= predict(SVMModel,test_data);
 for j=1:size(predict_label)
        a(j,1)=(isequal(predict_label(j,1),test_target(j,1)));
        test_label(j,1)=test_target(j,1);
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

[X,Y,T,AUC_SVM] = perfcurve(test_label,decision_values(:,2),'pos');
plot(X,Y,'g','linewidth',1)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by different classifers')
accuracy_SVM=sum(a(:))/size(a,1);
sensitivity_SVM=TP/(TP+FN);
specificity_SVM=TN/(TN+FP);
% auc_mrmr=plot_roc_V2(decision_values,label,'g');
hold on;
clear predict_label;clear test_label;clear decision_values;clear a;clear model;clear scores;
%% plot
hLe = legend({['KNN (AUC=' num2str(AUC_KNN) ')'],['LDA (AUC=' num2str(AUC_LDA)  ')'],['QDA (AUC=' num2str(AUC_QDA)  ')'],['SVM (AUC='  num2str(AUC_SVM)  ')']},...
    'location', 'southeast');
hLe.FontSize = 10;
saveas(gcf,'./RocTest.tif');

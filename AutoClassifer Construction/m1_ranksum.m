% U test to compare each of the 150 image featurs between 50 IU tRCCs and
% 50 matched IU ccRCCs

clear
tic

addpath('./tools')


load('./feats_train.mat', 'iu_I_IIIfeats', 'iu_IV_Vfeats');

load('./imFeaName.mat');

nFeas = numel(imFeaName);

pValue = zeros(nFeas, 1);
trHigh = zeros(nFeas, 1);
ratio = zeros(nFeas, 1);
for j = 1:nFeas
    pValue(j) = ranksum(iu_I_IIIfeats(:, j), iu_IV_Vfeats(:, j));
    if median(iu_IV_Vfeats(:, j)) > median(iu_I_IIIfeats(:, j))
        trHigh(j) = 1;
    end
    ratio(j) = median(iu_I_IIIfeats(:, j)) / median(iu_IV_Vfeats(:, j));
end    
tab = table(pValue, trHigh, ratio);
tab.feaInd = (1:nFeas)';
tab.feaName = imFeaName;
tab = sortrows(tab, 1);
[~, ~, ~, tab.pAdjust] = fdr_bh(tab.pValue, .05, 'pdep', 'yes');

tab = tab(:, [4, 5, 1, 6, 2, 3]);

tabSigOnly = tab(tab.pAdjust<0.05, :);
tabSigOnly = sortrows(tabSigOnly, 2);

save res_m1_ranksum tab tabSigOnly

toc

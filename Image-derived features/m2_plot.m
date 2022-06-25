% Scatter plot to show the adjusted p value and fold change of
% statistically significantly features. 

clear
close all

load('res_m1_ranksum.mat', 'tabSigOnly');
% load('imFeaName_reorder.mat');
load('imFeaName.mat');
[~, loc] = ismember(tabSigOnly.feaName, imFeaName);
[~, ind] = sort(loc);
tabSigOnly = tabSigOnly(ind, :);

n = size(tabSigOnly, 1);
foldChange = log2(tabSigOnly.ratio);
p = -log10(tabSigOnly.pAdjust);
y = (1:n)';

scatter(foldChange, y, 55, p, 'filled');
ylim([0, 76]);
yticks(y)
yticklabels(tabSigOnly.feaName);
xlabel('Log_2(fold change), fold change = Stages I-III median/Stages IV-V median');
ylabel('Image feature');
set(gca, 'TickLabelInterpreter', 'none');
grid on
colormap('jet');

% set(gcf,'PaperPositionMode','auto');
% print('res_m2_fig','-dtiff','-r300');

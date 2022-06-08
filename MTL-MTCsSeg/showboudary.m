se = strel('disk',5);
se2 = strel('disk',2);
mask=imresize(mask,[size(image,1) size(image,2)]);
    mask1=zeros(size(image,1),size(image,2));
    mask2=zeros(size(image,1),size(image,2));   
    mask3=zeros(size(image,1),size(image,2));
    mask4=zeros(size(image,1),size(image,2));
    mask1(find(mask==1))=1;
    mask2(find(mask==2))=1;
    mask3(find(mask==3))=1;
    mask4(find(mask==4))=1;
    
%% label1    
    mask1=imopen(mask1,se);
    mask1=imopen(mask1,se);
    %mask1=imdilate(mask1,se);
    %mask1=imdilate(mask1,se);
    [nuclei1,~,n1]=bwboundaries(mask1);
    %% label2  
    mask2=imopen(mask2,se);
    mask2=imopen(mask2,se);
%     mask2=imdilate(mask2,se);
%     mask2=imdilate(mask2,se);
[nuclei2,~,n2]=bwboundaries(mask2);
%% label3  
    mask3=imopen(mask3,se);
    mask3=imopen(mask3,se);
%     mask3=imdilate(mask3,se);
%     mask3=imdilate(mask3,se);
    [nuclei3,~,n3]=bwboundaries(mask3);
     %% label4  
    mask4=imopen(mask4,se);
    mask4=imopen(mask4,se);
%     mask4=imdilate(mask4,se);
%     mask4=imdilate(mask4,se);
    [nuclei4,~,n4]=bwboundaries(mask4);
    figure;imshow(image);hold on;

for k = 1:length(nuclei1)
        plot(nuclei1{k}(:,2), nuclei1{k}(:,1), 'g-', 'LineWidth', 2);
end

for k = 1:length(nuclei2)
        plot(nuclei2{k}(:,2), nuclei2{k}(:,1),'Color',[255 128 64]/255, 'LineWidth', 2);
end
for k = 1:length(nuclei3)
        plot(nuclei3{k}(:,2), nuclei3{k}(:,1), 'b-', 'LineWidth', 2);
end
for k = 1:length(nuclei4)
        plot(nuclei4{k}(:,2), nuclei4{k}(:,1), 'r-', 'LineWidth', 2);
end
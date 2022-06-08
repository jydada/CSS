clear;clc;

image_folder='D:\CSNet\dataset\DRIVE\train\images\';
label_folder='D:\CSNet\dataset\DRIVE\train\masks\';   
image_savepath='D:\CSNet\dataset\DRIVE\train\images_1\';%Label dataset
label_savepath='D:\CSNet\dataset\DRIVE\train\masks_1\';
content=dir([label_folder '*.tif']);
se = strel('disk',2);
cont1=1;
ws=256;
for m=1:size(content,1)
    image_name=content(m).name;
    image=imread([image_folder image_name(1:end-4) '.tif']); %读取原图
    image=image(:,:,[1 2 3]);
    im_x=size(image,1);
    im_y=size(image,2);
    %figure;imshow(image);
    mask=imread([label_folder image_name(1:end-4) '.tif']);  %读取标签
%     mask=im2bw(mask(:,:,2));
%     mask=imfill(mask,'hole');  
%      I = imerode(mask,se);
% %     I=imopen(I,se); 
%     
%     [x,y]=find(I==1);%
    %m=size(x);
    %采样数据总数
     amount=10;
     %amount = fix(m(1,1)*0.001);
    %随机生成采样点的坐标
    T = cell(amount,1);
    L = cell(amount,1);
    for ii=1:amount
        i=randi(im_x);
        j=randi(im_y);
         if ((i-ws>0&&j-ws>0)&&(i+ws<im_x&&j+ws<im_y))
            T{ii}=image(i-ws:i+ws-1,j-ws:j+ws-1,:);
            L{ii}=mask(i-ws:i+ws-1,j-ws:j+ws-1,:);
            imwrite(T{ii},[image_savepath image_name(1:end-4) '_' num2str(cont1) '.tif']);
            imwrite(L{ii},[label_savepath image_name(1:end-4) '_' num2str(cont1) '.tif']);
            cont1=cont1+1;
        end
    end
    
end
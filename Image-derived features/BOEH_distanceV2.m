clc;clear all;

imagepath='C:\Users\luhd\Desktop\1\';
maskpath='E:\png_masks\';


content=dir([imagepath,'*.tif']);

dSize=4;  %nuclear type
% -define edge patterns
[rs, cs] = find(triu(ones(dSize))); %triu()取值上三角函数 (所有类别的组合)
patterns = [rs, cs];
patterns = unique(patterns, 'rows');% C = unique(A)：返回的是和A中一样的值，但是没有重复元素。产生的结果向量按升序排序。
nPatt = size(patterns, 1);

nList = numel(content);
boeFiles = zeros(nList, nPatt);
Number = zeros(nList, dSize);
distance_feature = zeros(nList, nPatt*5);
area_feature = zeros(nList,dSize*5);
shape_feature = zeros(nList,3*dSize*5);
stain_feature = zeros(nList,3*dSize*5);
for i=1:size(content,1)
    fprintf('%d/%d to be Operated...\n',i ,size(content,1));
    t = tic;
    imagename=content(i).name;
    image=imread([imagepath imagename(1:end-4) '.tif']);
    mask=imread([maskpath imagename(1:end-4) '.tif']);
    props1=regionprops((mask==1),{'Area','Orientation','Centroid','MajorAxisLength','MinorAxisLength', 'BoundingBox'});
    centroids1 = zeros(numel(props1), 3, 'double');
    area1 = zeros(numel(props1), 1, 'double');
    major1=zeros(numel(props1), 1, 'double');
    minor1=zeros(numel(props1), 1, 'double');
    ratio1=zeros(numel(props1), 1, 'double');
    rMean1=zeros(numel(props1), 1, 'double');
    gMean1=zeros(numel(props1), 1, 'double');
    bMean1=zeros(numel(props1), 1, 'double');
    for j = 1:numel(props1)
        centroids1(j,[1 2]) = props1(j).Centroid;
        centroids1(j,3) = 1;
        area1(j,1)=props1(j).Area;
        major1(j,1)=props1(j).MajorAxisLength;
        minor1(j,1)=props1(j).MinorAxisLength;
        ratio1(j,1)=major1(j,1)/minor1(j,1);
%         rectangle('Position', props1(j).BoundingBox);
        try
            subImage = image(round(props1(j).BoundingBox(2):props1(j).BoundingBox(2)+props1(j).BoundingBox(4)),...
                round(props1(j).BoundingBox(1):props1(j).BoundingBox(1)+props1(j).BoundingBox(3)),[1 2 3]);
        catch
            warning('Problem using function.  Assigning a value of mean.');
            rMean1(j,1)=mean(rMean1,'all');
            gMean1(j,1)=mean(gMean1,'all');
            bMean1(j,1)=mean(bMean1,'all');
        end      
        rMean1(j,1)=mean(subImage(:,:,1),'all');
        gMean1(j,1)=mean(subImage(:,:,2),'all');
        bMean1(j,1)=mean(subImage(:,:,3),'all');
    end
    
    props2=regionprops((mask==2),{'Area','Orientation','Centroid','MajorAxisLength','MinorAxisLength', 'BoundingBox'});
    centroids2 = zeros(numel(props2), 3, 'double');
    area2 = zeros(numel(props2), 1, 'double');
    major2=zeros(numel(props2), 1, 'double');
    minor2=zeros(numel(props2), 1, 'double');
    ratio2=zeros(numel(props2), 1, 'double');
    rMean2=zeros(numel(props2), 1, 'double');
    gMean2=zeros(numel(props2), 1, 'double');
    bMean2=zeros(numel(props2), 1, 'double');
    for j = 1:numel(props2)
        centroids2(j,[1 2]) = props2(j).Centroid;
        centroids2(j,3) = 2;
        area2(j,1)=props2(j).Area;
        major2(j,1)=props2(j).MajorAxisLength;
        minor2(j,1)=props2(j).MinorAxisLength;
        ratio2(j,1)=major2(j,1)/minor2(j,1);
        try
        subImage = image(round(props2(j).BoundingBox(2):props2(j).BoundingBox(2)+props2(j).BoundingBox(4)),...
            round(props2(j).BoundingBox(1):props2(j).BoundingBox(1)+props2(j).BoundingBox(3)),[1 2 3]);
        catch
            warning('Problem using function.  Assigning a value of mean.');
            rMean2(j,1)=mean(rMean2,'all');
            gMean2(j,1)=mean(gMean2,'all');
            bMean2(j,1)=mean(bMean2,'all');
        end      
        rMean2(j,1)=mean(subImage(:,:,1),'all');
        gMean2(j,1)=mean(subImage(:,:,2),'all');
        bMean2(j,1)=mean(subImage(:,:,3),'all');
    end
    
    props3=regionprops((mask==3),{'Area','Orientation','Centroid','MajorAxisLength','MinorAxisLength', 'BoundingBox'});
    centroids3 = zeros(numel(props3), 3, 'double');
    area3 = zeros(numel(props3), 1, 'double');
    major3=zeros(numel(props3), 1, 'double');
    minor3=zeros(numel(props3), 1, 'double');
    ratio3=zeros(numel(props3), 1, 'double');
    rMean3=zeros(numel(props3), 1, 'double');
    gMean3=zeros(numel(props3), 1, 'double');
    bMean3=zeros(numel(props3), 1, 'double');
    for j = 1:numel(props3)
        centroids3(j,[1 2]) = props3(j).Centroid;
        centroids3(j,3) = 3;
        area3(j,1)=props3(j).Area;
        major3(j,1)=props3(j).MajorAxisLength;
        minor3(j,1)=props3(j).MinorAxisLength;
        ratio3(j,1)=major3(j,1)/minor3(j,1);
        try
            subImage = image(round(props3(j).BoundingBox(2):props3(j).BoundingBox(2)+props3(j).BoundingBox(4)),...
                round(props3(j).BoundingBox(1):props3(j).BoundingBox(1)+props3(j).BoundingBox(3)),[1 2 3]);
        catch
            warning('Problem using function.  Assigning a value of mean.');
            rMean3(j,1)=mean(rMean3,'all');
            gMean2(j,1)=mean(gMean3,'all');
            bMean3(j,1)=mean(bMean3,'all');
        end
        rMean3(j,1)=mean(subImage(:,:,1),'all');
        gMean3(j,1)=mean(subImage(:,:,2),'all');
        bMean3(j,1)=mean(subImage(:,:,3),'all');
    end
    
    props4=regionprops((mask==4),{'Area','Orientation','Centroid','MajorAxisLength','MinorAxisLength', 'BoundingBox'});
    centroids4 = zeros(numel(props4), 3, 'double');
    area4 = zeros(numel(props4), 1, 'double');
    major4=zeros(numel(props4), 1, 'double');
    minor4=zeros(numel(props4), 1, 'double');
    ratio4=zeros(numel(props4), 1, 'double');
    rMean4=zeros(numel(props4), 1, 'double');
    gMean4=zeros(numel(props4), 1, 'double');
    bMean4=zeros(numel(props4), 1, 'double');
    for j = 1:numel(props4)
        centroids4(j,[1 2]) = props4(j).Centroid;
        centroids4(j,3) = 4;
        area4(j,1)=props4(j).Area;
        major4(j,1)=props4(j).MajorAxisLength;
        minor4(j,1)=props4(j).MinorAxisLength;
        ratio4(j,1)=major4(j,1)/minor4(j,1);
        try
            subImage = image(round(props4(j).BoundingBox(2):props4(j).BoundingBox(2)+props4(j).BoundingBox(4)),...
                round(props4(j).BoundingBox(1):props4(j).BoundingBox(1)+props4(j).BoundingBox(3)),[1 2 3]);
        catch
            warning('Problem using function.  Assigning a value of mean.');
            rMean4(j,1)=mean(rMean4,'all');
            gMean4(j,1)=mean(gMean4,'all');
            bMean4(j,1)=mean(bMean4,'all');
        end
        rMean4(j,1)=mean(subImage(:,:,1),'all');
        gMean4(j,1)=mean(subImage(:,:,2),'all');
        bMean4(j,1)=mean(subImage(:,:,3),'all');
    end
    
    
 %% calculate the area of nuclei 
area_feature(i, :) =[mean(area1),std(area1),skewness(area1),kurtosis(area1),entropy(area1),...
     mean(area2),std(area2),skewness(area2),kurtosis(area2),entropy(area2),...
     mean(area3),std(area3),skewness(area3),kurtosis(area3),entropy(area3),...
     mean(area4),std(area4),skewness(area4),kurtosis(area4),entropy(area4),];
%% calculate the shape of nuclei 
shape_feature(i, :) =[mean(major1),std(major1),skewness(major1),kurtosis(major1),entropy(major1),...
    mean(minor1),std(minor1),skewness(minor1),kurtosis(minor1),entropy(minor1),...
    mean(ratio1),std(ratio1),skewness(ratio1),kurtosis(ratio1),entropy(ratio1),...
    mean(major2),std(major2),skewness(major2),kurtosis(major2),entropy(major2),...
    mean(minor2),std(minor2),skewness(minor2),kurtosis(minor2),entropy(minor2),...
    mean(ratio2),std(ratio2),skewness(ratio2),kurtosis(ratio2),entropy(ratio2),...
    mean(major3),std(major3),skewness(major3),kurtosis(major3),entropy(major3),...
    mean(minor3),std(minor3),skewness(minor3),kurtosis(minor3),entropy(minor3),...
    mean(ratio3),std(ratio3),skewness(ratio3),kurtosis(ratio3),entropy(ratio3),...
    mean(major4),std(major4),skewness(major4),kurtosis(major4),entropy(major4),...
    mean(minor4),std(minor4),skewness(minor4),kurtosis(minor4),entropy(minor4),...
    mean(ratio4),std(ratio4),skewness(ratio4),kurtosis(ratio4),entropy(ratio4),];
 %% calculate the staining of nuclei 
 stain_feature(i, :) =[mean(rMean1),std(rMean1),skewness(rMean1),kurtosis(rMean1),entropy(rMean1),...
     mean(gMean1),std(gMean1),skewness(gMean1),kurtosis(gMean1),entropy(gMean1),...
     mean(bMean1),std(bMean1),skewness(bMean1),kurtosis(bMean1),entropy(bMean1),...
     mean(rMean2),std(rMean2),skewness(rMean2),kurtosis(rMean2),entropy(rMean2),...
     mean(gMean2),std(gMean2),skewness(gMean2),kurtosis(gMean2),entropy(gMean2),...
     mean(bMean2),std(bMean2),skewness(bMean2),kurtosis(bMean2),entropy(bMean2),...
     mean(rMean3),std(rMean3),skewness(rMean3),kurtosis(rMean3),entropy(rMean3),...
     mean(gMean3),std(gMean3),skewness(gMean3),kurtosis(gMean3),entropy(gMean3),...
     mean(bMean3),std(bMean3),skewness(bMean3),kurtosis(bMean3),entropy(bMean3),...
     mean(rMean4),std(rMean4),skewness(rMean4),kurtosis(rMean4),entropy(rMean4),...
     mean(gMean4),std(gMean4),skewness(gMean4),kurtosis(gMean4),entropy(gMean4),...
     mean(bMean4),std(bMean4),skewness(bMean4),kurtosis(bMean4),entropy(bMean4),];


    
%% calculate number of nuclei edge patterns
centroids=[centroids1;centroids2;centroids3;centroids4];
DT = delaunayTriangulation(centroids(:,[1 2]));
tri = DT(:,:);
E = edges(DT);
ind=centroids(:,3);
E1 = ind(E);
E1 = sort(E1, 2);
[patts, ie, ip] = unique(E1, 'rows');
boeFiles(i, :) = hist(ip, 1:nPatt);   
%% calculate number of nuclei types
Number(i,:)=[numel(area1),numel(area2),numel(area3),numel(area4)];
    
%% calculate the distance of nuclei edges        
    distance1=[];distance2=[];distance3=[];distance4=[];distance5=[];distance6=[];distance7=[];distance8=[];distance9=[];distance10=[];
    cont1=1;cont2=1;cont3=1;cont4=1;cont5=1;cont6=1;cont7=1;cont8=1;cont9=1;cont10=1;
    for k = 1:length(E1)
        
        if E1(k,:)==[1,1]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            %distance1(cont1,1)=sqrt((centroids(E(k,1),1)-centroids(E(k,2),1)).^2+(centroids(E(k,1),2)-centroids(E(k,2),2)).^2);
            distance1(cont1,1)=norm(A-B);
            cont1=cont1+1;
        elseif E1(k,:)==[1,2]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance2(cont2,1)=norm(A-B);
            cont2=cont2+1;
        elseif E1(k,:)==[1,3]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance3(cont3,1)=norm(A-B);
            cont3=cont3+1;
        elseif E1(k,:)==[1,4]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance4(cont4,1)=norm(A-B);
            cont4=cont4+1;
        elseif E1(k,:)==[2,2]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance5(cont5,1)=norm(A-B);
            cont5=cont5+1;
        elseif E1(k,:)==[2,3]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance6(cont6,1)=norm(A-B);
            cont6=cont6+1;
        elseif E1(k,:)==[2,4]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance7(cont7,1)=norm(A-B);
            cont7=cont7+1;
        elseif E1(k,:)==[3,3]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance8(cont8,1)=norm(A-B);
            cont8=cont8+1;
        elseif E1(k,:)==[3,4]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance9(cont9,1)=norm(A-B);
            cont9=cont9+1;
        elseif E1(k,:)==[4,4]
            A=[centroids(E(k,1),1),centroids(E(k,1),2)];
            B=[centroids(E(k,2),1),centroids(E(k,2),2)];
            distance10(cont10,1)=norm(A-B);
            cont10=cont10+1;
        end
    end
    
    for m = 1:10
        name_string1 = ['distance' num2str(m)  ];
        %eval(name_string)
        if (isequal(eval(name_string1),[]))
            name_string2 = ['distance' num2str(m) '=0 '];
            eval(name_string2)
        end
        
    end
    
    
    distance_feature(i, :) =[mean(distance1),std(distance1),skewness(distance1),kurtosis(distance1),entropy(distance1),...
        mean(distance2),std(distance2),skewness(distance2),kurtosis(distance2),entropy(distance2),...
        mean(distance3),std(distance3),skewness(distance3),kurtosis(distance3),entropy(distance3),...
        mean(distance4),std(distance4),skewness(distance4),kurtosis(distance4),entropy(distance4),...
        mean(distance5),std(distance5),skewness(distance5),kurtosis(distance5),entropy(distance5),...
        mean(distance6),std(distance6),skewness(distance6),kurtosis(distance6),entropy(distance6),...
        mean(distance7),std(distance7),skewness(distance7),kurtosis(distance7),entropy(distance7),...
        mean(distance8),std(distance8),skewness(distance8),kurtosis(distance8),entropy(distance8),...
        mean(distance9),std(distance9),skewness(distance9),kurtosis(distance9),entropy(distance9),...
        mean(distance10),std(distance10),skewness(distance10),kurtosis(distance10),entropy(distance10),];
    fprintf('%d/%d finished, time %f\n', i, nList, toc(t));
    
    
end
      distance_feature(isnan(distance_feature)) = 0 ;
%      feats=normalize(distance_feature,1,'range');
feats=[area_feature,shape_feature,stain_feature,Number,boeFiles,distance_feature];

 
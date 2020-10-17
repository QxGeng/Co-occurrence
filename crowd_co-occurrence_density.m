clear;

load('./mat/original_pre_boxes.mat');
load('./dataset/Crowd Face/ground_truth/density.mat');

boxes = crowds_face_boxes_originalhr1;
density_info = density_data;
prob_thresh = 0.5;
a=-log((1/0.03)-1);    
b=-log((1/prob_thresh)-1);  
high_score=-log((1/0.1)-1);        


for k = 1:size(boxes,1)
    boxes{k}(:,1:4) = round(boxes{k}(:,1:4));
    boxes{k}(:,3) = boxes{k}(:,3) - boxes{k}(:,1);
    boxes{k}(:,4) = boxes{k}(:,4) - boxes{k}(:,2);
end

area = boxes;

for i = 1:size(boxes,1)
    area{i}(:,1)=area{i}(:,3) .* area{i}(:,4);  
    area{i}(:,2)=area{i}(:,end);    
    
    boxes{i}(:,3) = boxes{i}(:,1) + boxes{i}(:,3);
    boxes{i}(:,4) = boxes{i}(:,2) + boxes{i}(:,4);
    
    x1 = density_info{i}(1,1);   
    x2 = density_info{i}(1,3);   
    x3 = density_info{i}(2,3);   
    x4 = density_info{i}(3,3);   
    y1 = density_info{i}(1,2);  
    y2 = density_info{i}(1,4);  
    y3 = density_info{i}(4,4);   
    y4 = density_info{i}(7,4);   
        
    for j = 1:size(boxes{i},1)
        if boxes{i}(j,1)>=x1 && boxes{i}(j,2)>=y1 && boxes{i}(j,3)<x2 && boxes{i}(j,4)<=y2 
            area{i}(j,3)=1;             
        elseif boxes{i}(j,1)>=x2 && boxes{i}(j,2)>=y1 && boxes{i}(j,3)<x3 && boxes{i}(j,4)<=y2 
            area{i}(j,3)=2;             
        elseif boxes{i}(j,1)>=x3 && boxes{i}(j,2)>=y1 && boxes{i}(j,3)<x4 && boxes{i}(j,4)<=y2
            area{i}(j,3)=3;               
        elseif boxes{i}(j,1)>=x1 && boxes{i}(j,2)>=y2 && boxes{i}(j,3)<x2 && boxes{i}(j,4)<=y3 
            area{i}(j,3)=4;               
        elseif boxes{i}(j,1)>=x2 && boxes{i}(j,2)>=y2 && boxes{i}(j,3)<x3 && boxes{i}(j,4)<=y3
            area{i}(j,3)=5;               
        elseif boxes{i}(j,1)>=x3 && boxes{i}(j,2)>=y2 && boxes{i}(j,3)<x4 && boxes{i}(j,4)<=y3
            area{i}(j,3)=6;                
        elseif boxes{i}(j,1)>=x1 && boxes{i}(j,2)>=y3 && boxes{i}(j,3)<x2 && boxes{i}(j,4)<=y4
            area{i}(j,3)=7;                
        elseif boxes{i}(j,1)>=x2 && boxes{i}(j,2)>=y3 && boxes{i}(j,3)<x3 && boxes{i}(j,4)<=y4 
            area{i}(j,3)=8;                       
        elseif boxes{i}(j,1)>=x3 && boxes{i}(j,2)>=y3 && boxes{i}(j,3)<x4 && boxes{i}(j,4)<=y4 
            area{i}(j,3)=9;     
        end
    end
end

for m = 1:size(area,1)
    num1=0;sum1=0; 
    num2=0;sum2=0;
    num3=0;sum3=0;
    num4=0;sum4=0;
    num5=0;sum5=0;
    num6=0;sum6=0;
    num7=0;sum7=0;
    num8=0;sum8=0;
    num9=0;sum9=0;
    for n = 1:size(area{m},1)
        if area{m}(n,3)==1
            if area{m}(n,2)>=high_score
                num1 = num1 + 1;
                sum1 = sum1 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==2
            if area{m}(n,2)>=high_score
                num2 = num2 + 1;
                sum2 = sum2 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==3
            if area{m}(n,2)>=high_score
                num3 = num3 + 1;
                sum3 = sum3 + area{m}(n,1);
            end
        end
        if density_info{m}(4,5)<=density_thresh && area{m}(n,3)==4
            if area{m}(n,2)>=high_score
                num4 = num4 + 1;
                sum4 = sum4 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==5
            if area{m}(n,2)>=high_score
                num5 = num5 + 1;
                sum5 = sum5 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==6
            if area{m}(n,2)>=high_score
                num6 = num6 + 1;
                sum6 = sum6 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==7
            if area{m}(n,2)>=high_score
                num7 = num7 + 1;
                sum7 = sum7 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==8
            if area{m}(n,2)>=high_score
                num8 = num8 + 1;
                sum8 = sum8 + area{m}(n,1);
            end
        end
        if area{m}(n,3)==9
            if area{m}(n,2)>=high_score
                num9 = num9 + 1;
                sum9 = sum9 + area{m}(n,1);
            end
        end
    end
    face_size_avg1 = sum1 / num1;
    min1 = round(0.8 * face_size_avg1);
    max1 = round(1.2 * face_size_avg1);    
    face_size_avg2 = sum2 / num2;
    min2 = round(0.8 * face_size_avg2);
    max2 = round(1.2 * face_size_avg2);
    face_size_avg3 = sum3 / num3;
    min3 = round(0.8 * face_size_avg3);
    max3 = round(1.2 * face_size_avg3);   
    face_size_avg4 = sum4 / num4;
    min4 = round(0.8 * face_size_avg4);
    max4 = round(1.2 * face_size_avg4);    
    face_size_avg5 = sum5 / num5;
    min5 = round(0.8 * face_size_avg5);
    max5 = round(1.2 * face_size_avg5);    
    face_size_avg6 = sum6 / num6;
    min6 = round(0.8 * face_size_avg6);
    max6 = round(1.2 * face_size_avg6);    
    face_size_avg7 = sum7 / num7;
    min7 = round(0.8 * face_size_avg7);
    max7 = round(1.2 * face_size_avg7);    
    face_size_avg8 = sum8 / num8;
    min8 = round(0.8 * face_size_avg8);
    max8 = round(1.2 * face_size_avg8);   
    face_size_avg9 = sum9 / num9;
    min9 = round(0.8 * face_size_avg9);
    max9 = round(1.2 * face_size_avg9);   
    for o = 1:size(area{m},1)
        if area{m}(o,3)==1 && min1<=area{m}(o,1) && area{m}(o,1)<=max1
            rou = 1/density_info{m}(1,5);
            w =  1/(1 + exp(-(rou * num1)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==2 && min2<=area{m}(o,1) && area{m}(o,1)<=max2
            rou = 1/density_info{m}(2,5);
            w =  1/(1 + exp(-(rou * num2)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==3 && min3<=area{m}(o,1) && area{m}(o,1)<=max3
            rou = 1/density_info{m}(3,5);
            w =  1/(1 + exp(-(rou * num3)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==4 && min4<=area{m}(o,1) && area{m}(o,1)<=max4
            rou = 1/density_info{m}(4,5);
            w =  1/(1 + exp(-(rou * num4)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==5 && min5<=area{m}(o,1) && area{m}(o,1)<=max5
            rou = 1/density_info{m}(5,5);
            w =  1/(1 + exp(-(rou * num5)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==6 && min6<=area{m}(o,1) && area{m}(o,1)<=max6
            rou = 1/density_info{m}(6,5);
            w =  1/(1 + exp(-(rou * num6)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==7 && min7<=area{m}(o,1) && area{m}(o,1)<=max7
            rou = 1/density_info{m}(7,5);
            w =  1/(1 + exp(-(rou * num7)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==8 && min8<=area{m}(o,1) && area{m}(o,1)<=max8
            rou = 1/density_info{m}(8,5);
            w =  1/(1 + exp(-(rou * num8)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        if area{m}(o,3)==9 && min9<=area{m}(o,1) && area{m}(o,1)<=max9
            rou = 1/density_info{m}(9,5);
            w =  1/(1 + exp(-(rou * num9)));
            if area{m}(o,2)<=b
                area{m}(o,2) = w * area{m}(o,2)  + area{m}(o,2);
            end
        end
        boxes{m}(:,5)=area{m}(:,2);
    end
end

for i = 1:size(boxes,1)
    l1 = size(boxes{i},1);
    j = 1;
    while j <= l1
        if boxes{i}(j,end)<a
            boxes{i}(j,:)=[];
            l1=l1-1;
        else
            j = j+1;
        end
    end  
end

save('./mat/crowd_coexistence_density','boxes')




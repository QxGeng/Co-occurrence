function visualize2_detection(bbox,thr)

% color=[1 0 0];    %红色
% color=[0 0 1];         %蓝色
% color=[0.67 0 1];         %天蓝色
% color=[0 1 0];         %绿色
% color=[1 0 1];                  %洋红
% color=[1 0.41176 0.70588];   %hotpink

color=[0 1 1];         %青色

scores = vl_nnsigmoid(bbox(:,end));

hold on;
for i = 1:size(bbox, 1)
  if scores(i) < thr, continue; end
  %color = colors(ceil(100*scores(i)), :);0
  bw = bbox(i,3) - bbox(i,1) + 1;
  bh = bbox(i,4) - bbox(i,2) + 1;
  if min([bw bh]) <= 20
    lw = 1;
  else
    lw = max(2, min(3, min([bh/20, bw/20])));
  end
  lw = lw * scores(i); 
%   rectangle('position', [bbox(i,1:2) bbox(i,3:4)-bbox(i,1:2)+1], ...
%             'EdgeColor', color, 'LineWidth', 0.5,'Curvature',[1,1]);
rectangle('position', [bbox(i,1:2) bbox(i,3:4)-bbox(i,1:2)+1], ...
            'EdgeColor', color, 'LineWidth', lw,'Curvature',[1,1]);
end
hold on;
axis off;

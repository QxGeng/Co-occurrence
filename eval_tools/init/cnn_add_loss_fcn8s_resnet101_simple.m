function net = cnn_add_loss_fcn8s_resnet101_simple(opts, net)

%% 冻结层参数，那么这些层在训练时，就不会更新权重
if opts.freezeResNet,
    for i = 1:numel(net.params)
        net.params(i).learningRate = 0;
    end
end

%%   
N = opts.clusterNum;
skipLRMultipliers = opts.skipLRMult;
learningRates = skipLRMultipliers;     %学习率[1, 0.001, 0.0001, 0.00001];

%% remove prob    移除原网络的‘prob’层
if ~isnan(net.getLayerIndex('prob'))
    net.removeLayer('prob');
end

% remove *res5*,bn5,pool5,fc1000 layers
% 移除与名字中有‘res5’字样的层，以及bn5层，pool5层和fc1000层
names = {};
for i = 1:numel(net.layers)
    if ~isempty(strfind(net.layers(i).name,'res5')) || ...
            ~isempty(strfind(net.layers(i).name, 'bn5'))
        names{end+1} = net.layers(i).name; 
    end
end
names{end+1} = 'pool5'; 
names{end+1} = 'fc1000';

for i = 1:numel(names)
    net.removeLayer(names{i});
end

% NOTE: we end up not using features from res5 
% %% update 'fc1000' (on 'pool5')
% lidx = net.getLayerIndex('fc1000');
% fidx = net.getParamIndex('fc1000_filter'); 
% bidx = net.getParamIndex('fc1000_bias'); 
% v = net.params(fidx).value; 
% [h,w,in,~] = size(v); 
% out = 5*N; 
% net.params(fidx).value = zeros(h,w,in,out,'single');
% net.params(fidx).learningRate = learningRates(1);
% net.params(bidx).value = zeros(1,out,'single');
% net.params(bidx).learningRate = learningRates(1);
% net.layers(lidx).block.size(4) = out;
% 
% %% add upsampling 
% filter = single(bilinear_u(4, 1, 5*N));
% ctblk = dagnn.ConvTranspose('size', size(filter), 'upsample', 2, ...
%                             'crop', [0, 0, 0, 0], 'hasBias', false);
% net.addLayer('score2', ctblk, 'fc1000', 'score2', 'score2f');
% fidx = net.getParamIndex('score2f'); 
% net.params(fidx).value = filter; 
% net.params(fidx).learningRate = 0;


%% add predictors on 'res4b22x'  为'res4b22x'层添加一个预测层（卷积层）
filter = zeros(1,1,1024,5*N,'single');    %%这个参数什么意思呢  zeros()产生一个全0矩阵
bias = zeros(1,5*N,'single');
cblk = dagnn.Conv('size',size(filter),'stride',1,'pad',0);   %size返回数组的尺寸（每一维的大小）
%层名字为“score_res4”，类型为cblk，输入变量为'res4b22x'，输出变量为'score_res4'
%层参数名为'score_res4_filter'和'score_res4_bias'
net.addLayer('score_res4', cblk, 'res4b22x', 'score_res4', ...
             {'score_res4_filter', 'score_res4_bias'});
fidx = net.getParamIndex('score_res4_filter'); 
bidx = net.getParamIndex('score_res4_bias'); 
net.params(fidx).value = filter;     %设定层参数
net.params(fidx).learningRate = learningRates(2); 
net.params(bidx).value = bias; 
net.params(bidx).learningRate = learningRates(2);

%% add upsampling 增添上采样层
filter = single(bilinear_u(4, 1, 5*N));    %single 把任何数据类型转为单精度数

% adapt for different input sizes (we end up using 500x500)
% 上采样层的参数根据input sizes（500x500）而定。
if all(opts.inputSize==500)
    ctblk = dagnn.ConvTranspose('size', size(filter), 'upsample', 2, ...
                                'crop', [1,2,1,2], 'hasBias', false);
elseif opts.inputSize(1)==750 && opts.inputSize(2)==1000
    ctblk = dagnn.ConvTranspose('size', size(filter), 'upsample', 2, ...
                                'crop', [1,1,1,2], 'hasBias', false);
elseif opts.inputSize(1)==300 && opts.inputSize(2)==300
    ctblk = dagnn.ConvTranspose('size', size(filter), 'upsample', 2, ...
                                'crop', [1,1,1,1], 'hasBias', false);
else
    error('Input size not supported');
end

% define bilinear interpolation filter (fixed weights) （固定权重）
net.addLayer('score4', ctblk, 'score_res4', 'score4', 'score4f');
fidx = net.getParamIndex('score4f');
net.params(fidx).value = filter;
net.params(fidx).learningRate = 0;

%% add predictors on 'res3dx'
filter = zeros(1,1,512,5*N,'single');
bias = zeros(1,5*N,'single');
cblk = dagnn.Conv('size',size(filter),'stride',1,'pad',0);
net.addLayer('score_res3', cblk, 'res3b3x', 'score_res3', ...
             {'score_res3_filter', 'score_res3_bias'});
fidx = net.getParamIndex('score_res3_filter'); 
bidx = net.getParamIndex('score_res3_bias'); 
net.params(fidx).value = filter;
net.params(fidx).learningRate = learningRates(3);
net.params(bidx).value = bias; 
net.params(bidx).learningRate = learningRates(3);

% Note: since we train on cropped regions with fixed size,
% we don't need to add cropping layers for aligning heat maps
% before adding them

% sum 
net.addLayer('fusex',dagnn.Sum(),{'score_res3', 'score4'}, ...
             'score_res3_fused');

%% rename last score to score_final 
net.renameVar('score_res3_fused', 'score_final');

%
net.addLayer('split', dagnn.Split('childIds', {1:N, N+1:5*N}), ...
             'score_final', {'score_cls', 'score_reg'});

% only use customized loss when we have variable sample size
net.addLayer('loss_cls', dagnn.Loss('loss', 'logistic'), ...
             {'score_cls', 'label_cls'}, 'loss_cls');
net.addLayer('loss_reg', dagnn.HuberLoss(), ...
             {'score_reg', 'label_reg', 'label_cls'}, 'loss_reg');


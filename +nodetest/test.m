clc;
clear;
close all;
rng default;

% % 加载数据 并简化 32*24*3通道*4样本 存档
% load data/XY2.mat TRAIN*
% TRAINX = TRAINX(:,:,:,1:4);
% TRAINY = TRAINY(:,:,:,1:4);
% TRAINXOLD = TRAINX;
% TRAINX = zeros(32,24,1,size(TRAINXOLD,4));
% for i=1:size(TRAINXOLD,4)
%     TRAINX(:,:,1,i) = imresize(TRAINXOLD(:,:,1,i), [32 24]);
% end
% TRAINX = repmat(TRAINX, [1 1 3 1]);
% TRAINX = TRAINX + randn(size(TRAINX)) * 0.1;
% color = TRAINX(:,:,:,1) /2 +0.5;
% % imshow(color);
% % return
% clear TRAINXOLD color i;
% save data/XY2F.mat;
% return;

% 加载简化后的数据
load data/XY2F.mat

% 快速版网络 测试FF BP用
% 用随机初始化幅度将激活值控制在稳定区
MINIBATCHSIZE = 4;
%     TRAINX = repmat(TRAINX, [1 1 1 10]);% MINIBATCH复制10倍
%     TRAINY = repmat(TRAINY, [1 1 1 10]);% MINIBATCH复制10倍
%     MINIBATCHSIZE = MINIBATCHSIZE * 10;% MINIBATCH复制10倍
x = node.Data(TRAINX, MINIBATCHSIZE);
y = node.Data(TRAINY, MINIBATCHSIZE);
c0 = node.Conv(x, 3, 5);        c0.P = randn(size(c0.P));
c1 = node.ConvParallel(c0, 3, 10);   c1.P = randn(size(c1.P))/10;
b2 = node.Bias(c1);             b2.P = randn(size(b2.P))/3;
t3 = node.Tanh(b2);
s4 = node.MaxPool(t3);
flat = node.Flatten(s4);
f6 = node.FullCon(flat, 6);     f6.P = randn(size(f6.P))/40;
b7 = node.Bias(f6);             b7.P = randn(size(b7.P))/3;
d8 = node.DropoutTrain(b7);
smn = node.SoftMaxNorm(d8);
sml = node.SoftMaxLoss(smn, y);

% disp('------------------------------------------------------------------------------------------------')
% fig1 = figure('Name','O','NumberTitle','off');
% fig2 = figure('Name','P','NumberTitle','off');
% fig3 = figure('Name','gradO','NumberTitle','off');
% fig4 = figure('Name','gradP','NumberTitle','off');
% for i=1:10
%     % ff
%     t = tic;
%     sml.ff();
%     toc(t);
%     
%     % bp
%     t = tic;
%     sml.bp();
%     toc(t);
%     fprintf('loss: %e\n', sml.O);
%     
%     % 
%     sml.debugHist(fig1, fig2, fig3, fig4, 4, 5, 0);
%     drawnow;
%     
%     % update
%     LR = 0.01;
%     c0.P = c0.P - LR*c0.gradP;
%     c1.P = c1.P - LR*c1.gradP;
%     b2.P = b2.P - LR*b2.gradP;
%     f6.P = f6.P - LR*f6.gradP;
%     b7.P = b7.P - LR*b7.gradP;
% end

% 最后一次ff
disp('------------------------------------------------------------------------------------------------')
t = tic;
sml.ff();
toc(t);

% check ff
nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, f6.P, b7.P, d8.mask, x.O, c0.O, t3.O, s4.O, flat.O, b7.O, d8.O, smn.O, y.O, sml.O);

% 最后一次bp
disp('------------------------------------------------------------------------------------------------')
t = tic;
sml.bp();
toc(t);

% check bp
% 验证smn sml
% 换另一种SoftMatWithLoss的算法来验证 http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
checkVal = zeros(1,1,6,MINIBATCHSIZE);
for i=1:MINIBATCHSIZE
    vec = permute(d8.O(:,:,:,i), [3 4 1 2]);
    checkVal(:,:,:,i) = permute( exp(vec)/sum(exp(vec)), [3 4 1 2]);
end
lbl = TRAINY(:,:,:,1:MINIBATCHSIZE);% minibatch偏移
lbl = permute(lbl, [3 4 1 2]);
[~, lbl] = max(lbl);
idx = sub2ind(size(checkVal), ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), lbl, 1:MINIBATCHSIZE);
checkVal(idx) = checkVal(idx)-1;
checkVal = checkVal / MINIBATCHSIZE;
norm(reshape(checkVal-d8.gradO,[],1))

% 梯度校验验证fullcon的参数的梯度
eps = 1e-6;
gradP = zeros([6 16*12*10]);
util.dispstat('','init');
for i=1:16*12*10
    util.dispstat(sprintf('Processing %d%%', round(i/(16*12*10)*100))); 
    for j=1:6
        PL = f6.P;
        PL(j,i) = PL(j,i)-eps;
        LL = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, PL, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
        PR = f6.P;
        PR(j,i) = PR(j,i)+eps;
        LR = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, PR, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
        gradP(j,i) = (LR-LL) / 2 / eps;
    end
end
f6.gradP(1:6,1:6)
gradP(1:6,1:6)
f6.gradP(end-5:end,end-5:end)
gradP(end-5:end,end-5:end)
norm(f6.gradP(:) - gradP(:))

% 梯度校验验证bias的参数的梯度
eps = 1e-6;
gradP = zeros([6 1]);
for i=1:6
    PL = b7.P;
    PL(i) = PL(i)-eps;
    LL = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, f6.P, PL, d8.mask, [], [], [], [], [], [], [], [], [], []);
    PR = b7.P;
    PR(i) = PR(i)+eps;
    LR = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, f6.P, PR, d8.mask, [], [], [], [], [], [], [], [], [], []);
    gradP(i) = (LR-LL) / 2 / eps;
end
b7.gradP
gradP
norm(b7.gradP(:) - gradP(:))

% 梯度校验验证bias的参数的梯度
eps = 1e-6;
gradP = zeros([10 1]);
for i=1:10
    PL = b2.P;
    PL(i) = PL(i)-eps;
    LL = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, PL, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    PR = b2.P;
    PR(i) = PR(i)+eps;
    LR = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, c1.P, PR, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    gradP(i) = (LR-LL) / 2 / eps;
end
b2.gradP
gradP
norm(b2.gradP(:) - gradP(:))

% 梯度校验验证x的激活的梯度
eps = 1e-6;
gradP = zeros([32 24 3 4]);
util.dispstat('','init');
for i=1:numel(gradP)
    util.dispstat(sprintf('Processing %d%%', round(i/numel(gradP)*100))); 
    PL = TRAINX;
    PL(i) = PL(i)-eps;
    LL = nodetest.checkff(PL, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    PR = TRAINX;
    PR(i) = PR(i)+eps;
    LR = nodetest.checkff(PR, TRAINY, MINIBATCHSIZE, c0.P, c1.P, b2.P, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    gradP(i) = (LR-LL) / 2 / eps;
end
x.gradO(1:10,1:10,1,1)
gradP(1:10,1:10,1,1)
x.gradO(1:10,1:10,end,end)
gradP(1:10,1:10,end,end)
norm(x.gradO(:) - gradP(:))

% 梯度校验验证conv的参数的梯度
eps = 1e-6;
gradP = zeros([3 3 5 10]);
for i=1:numel(gradP)
    PL = c1.P;
    PL(i) = PL(i)-eps;
    LL = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, PL, b2.P, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    PR = c1.P;
    PR(i) = PR(i)+eps;
    LR = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0.P, PR, b2.P, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    gradP(i) = (LR-LL) / 2 / eps;
end
c1.gradP(:,:,1,1)
gradP(:,:,1,1)
c1.gradP(:,:,end,end)
gradP(:,:,end,end)
norm(c1.gradP(:) - gradP(:))

% 梯度校验验证conv的参数的梯度
eps = 1e-6;
gradP = zeros([3 3 3 5]);
for i=1:numel(gradP)
    PL = c0.P;
    PL(i) = PL(i)-eps;
    LL = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, PL, c1.P, b2.P, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    PR = c0.P;
    PR(i) = PR(i)+eps;
    LR = nodetest.checkff(TRAINX, TRAINY, MINIBATCHSIZE, PR, c1.P, b2.P, f6.P, b7.P, d8.mask, [], [], [], [], [], [], [], [], [], []);
    gradP(i) = (LR-LL) / 2 / eps;
end
c0.gradP(:,:,1,1)
gradP(:,:,1,1)
c0.gradP(:,:,end,end)
gradP(:,:,end,end)
norm(c0.gradP(:) - gradP(:))

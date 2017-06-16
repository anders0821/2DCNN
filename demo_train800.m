clc;
clear;
close all;
drawnow;
rng default;

% 算法超参数
MINIBATCHSIZE = 100;
LR = 0.01;
WD = 0.0005;
MOM = 0.0;

% 程序参数
DISPLAY_LOSSES_ACCS_INTERVAL = round(1000/MINIBATCHSIZE);
DISPLAY_HIST_INTERVAL = 1e12;
SNAPSHOT_INTERVAL = round(1000/MINIBATCHSIZE);
RESUME_FROM_SNAPSHOT_ITER = 0;
MAX_ITER = 1e12;

% 加载数据
load data/XY2_SMALLER_NOMEAN.mat

% 初始化网络
x = node.Data(TRAINX, MINIBATCHSIZE);
y = node.Data(TRAINY, MINIBATCHSIZE);
xg = node.GaborParallel(x, 29, 1, 8);

c11 = node.ConvParallelStopBp(xg, 7, 30);      c11.P = randn(size(c11.P)) * 0.01*600;
b11 = node.Bias(c11);
t11 = node.Tanh(b11);
s1 = node.MaxPool(t11);
d1 = node.DropoutTrain(s1);

c21 = node.ConvParallel(d1, 5, 40);     c21.P = randn(size(c21.P)) * 0.01*10;
b21 = node.Bias(c21);
t21 = node.Tanh(b21);
s2 = node.MaxPool(t21);
d2 = node.DropoutTrain(s2);

flat = node.Flatten(d2);

% 最后一个全连接层 表示prob 不用tanh归一化用SoftMax归一化 即SoftMaxNorm(W*x+b)
f7 = node.FullCon(flat, 6);       f7.P = randn(size(f7.P)) * 0.01*2;
b7 = node.Bias(f7);
smn = node.SoftMaxNorm(b7);

% 交叉熵损失 输入prob与label计算标量损失
sml = node.SoftMaxLoss(smn, y);

% debugMemory
[totalMemory, totalParamMemory] = sml.debugMemory();
totalMemory = totalMemory*8/1024/1024/1204;
totalParamMemory = totalParamMemory*8/1024/1024/1204;
fprintf('totalMemory: %f GB\ntotalParamMemory: %f GB\n', totalMemory, totalParamMemory);

% % 剖析ff bp时间
% disp('------------------------------------------------------------------------------------------------')
% profile on;
% tic;
% for i=1:10
%     sml.ff();
%     sml.bp();
% end
% toc;
% profile off;
% profile viewer;
% return;

disp('------------------------------------------------------------------------------------------------')
% 学习过程中solver管理的状态量
losses = cell(0);
accs = cell(0);
v = cell(0);
v{1} =  zeros(size(c11.gradP));
v{2} =  zeros(size(b11.gradP));
v{3} =  zeros(size(c21.gradP));
v{4} =  zeros(size(b21.gradP));
v{5} =  zeros(size( f7.gradP));
v{6} =  zeros(size( b7.gradP));

% 如果需要则从快照中恢复
if(RESUME_FROM_SNAPSHOT_ITER>0)
    fn = ['snapshot/snapshot-' num2str(RESUME_FROM_SNAPSHOT_ITER) '.mat'];
    load(fn, 'params', 'losses', 'accs', 'v');
    fprintf('load from %s\n', fn);
    sml.setPs(params);
end

% 主迭代
for iter=(RESUME_FROM_SNAPSHOT_ITER+1):MAX_ITER
    t = tic;
    
    % ff
    sml.ff();
    
    % loss
    loss = sml.O;
    losses{iter} = loss;
    
    % acc
    [~,lbl] = max(permute(smn.O, [3 4 1 2]));
    [~, groundtruthLbl] = max(permute(y.O, [3 4 1 2]));
    acc = sum(lbl==groundtruthLbl) / MINIBATCHSIZE;
    accs{iter} = acc;
    
    % display losses accs
    fprintf('iter: %d, loss: %e, acc: %.3f %%\n', iter, loss, acc*100);
    fprintf('%d -> %d\n', [groundtruthLbl; lbl]);
    if(mod(iter,DISPLAY_LOSSES_ACCS_INTERVAL)==0)
        if(~(exist('fig0', 'var') && ishghandle(fig0)))
            fig0 = figure('Name','losses accs','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
        end
        set(0, 'CurrentFigure', fig0);
        subplot(1,2,1);
        semilogy((1:iter)*MINIBATCHSIZE, cell2mat(losses), '+');
        title('losses');
        subplot(1,2,2);
        plot((1:iter)*MINIBATCHSIZE, cell2mat(accs)*100, '+');
        ylim([0 100]);
        title('accs');
        drawnow;
    end
    
    % bp
    sml.bp();
    
    % debugHist
    if(mod(iter,DISPLAY_HIST_INTERVAL)==0)
        if(~(exist('fig1', 'var') && ishghandle(fig1)))
            fig1 = figure('Name','O','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
        end
        if(~(exist('fig2', 'var') && ishghandle(fig2)))
            fig2 = figure('Name','P','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
        end
        if(~(exist('fig3', 'var') && ishghandle(fig3)))
            fig3 = figure('Name','gradO','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
        end
        if(~(exist('fig4', 'var') && ishghandle(fig4)))
            fig4 = figure('Name','gradP','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
        end
        sml.debugHist(fig1, fig2, fig3, fig4, 8, 7, 0);
        drawnow;
    end
    
    % update
    v{1} = MOM*v{1} + (1-MOM)*LR*(-c11.gradP-WD*c11.P);
    v{2} = MOM*v{2} + (1-MOM)*LR*(-b11.gradP);
    v{3} = MOM*v{3} + (1-MOM)*LR*(-c21.gradP-WD*c21.P);
    v{4} = MOM*v{4} + (1-MOM)*LR*(-b21.gradP);
    v{5} = MOM*v{5} + (1-MOM)*LR*(- f7.gradP-WD* f7.P);
    v{6} = MOM*v{6} + (1-MOM)*LR*(- b7.gradP);
    
    c11.P = c11.P + v{1};
    b11.P = b11.P + v{2};
    c21.P = c21.P + v{3};
    b21.P = b21.P + v{4};
    f7.P  =  f7.P + v{5};
    b7.P  =  b7.P + v{6};
    
    % snapshot
    if(mod(iter,SNAPSHOT_INTERVAL)==0)
        fn = ['snapshot/snapshot-' num2str(iter) '.mat'];
        fprintf('save to %s\n', fn);
        params = sml.getPs();
        save(fn, 'params', 'losses', 'accs', 'v');
    end
    
    toc(t);
end

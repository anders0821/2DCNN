clc;
clear;
close all;
drawnow;
rng default;

for RESUME_FROM_SNAPSHOT_ITER = [490:-10:100]
    
    % 算法超参数
    MINIBATCHSIZE = 4;
    MAX_ITER = RESUME_FROM_SNAPSHOT_ITER+(1548+200+200)/MINIBATCHSIZE;
    
    % 加载数据
    load data/XY2_SMALLER_NOMEAN.mat
    
    % 初始化网络
    x = node.Data(TRAINX, MINIBATCHSIZE);
    y = node.Data(TRAINY, MINIBATCHSIZE);
    xg = node.GaborParallel(x, 29, 1, 8);
    
    c11 = node.ConvParallelStopBp(xg, 7, 30);      c11.P = randn(size(c11.P)) * 0.01*400;
    b11 = node.Bias(c11);
    t11 = node.Tanh(b11);
    s1 = node.MaxPool(t11);
    d1 = node.DropoutTest(s1);
    
    c21 = node.ConvParallel(d1, 5, 40);     c21.P = randn(size(c21.P)) * 0.01*8;
    b21 = node.Bias(c21);
    t21 = node.Tanh(b21);
    s2 = node.MaxPool(t21);
    d2 = node.DropoutTest(s2);
    
    flat = node.Flatten(d2);
    
    % 最后一个全连接层 表示prob 不用tanh归一化用SoftMax归一化 即SoftMaxNorm(W*x+b)
    f7 = node.FullCon(flat, 6);       f7.P = randn(size(f7.P)) * 0.01*2;
    b7 = node.Bias(f7);
    smn = node.SoftMaxNorm(b7);
    
    % 交叉熵损失 输入prob与label计算标量损失
    sml = node.SoftMaxLoss(smn, y);
    
    % 学习过程中solver管理的状态量
    accs = cell(0);
    
    % 如果需要则从快照中恢复
    if(RESUME_FROM_SNAPSHOT_ITER>0)
        fn = ['snapshot/snapshot-' num2str(RESUME_FROM_SNAPSHOT_ITER) '.mat'];
        load(fn, 'params', 'losses', 'accs', 'v');
        sml.setPs(params);
    end
    
    accs = cell(0);
    
    confusePair = zeros(2, (MAX_ITER-RESUME_FROM_SNAPSHOT_ITER)*4);
    
    % 主迭代
    for iter=(RESUME_FROM_SNAPSHOT_ITER+1):MAX_ITER
        % ff
        sml.ff();
        
        % acc
        [~,lbl] = max(permute(smn.O, [3 4 1 2]));
        [~, groundtruthLbl] = max(permute(y.O, [3 4 1 2]));
        acc = sum(lbl==groundtruthLbl) / MINIBATCHSIZE;
        accs{iter} = acc;
        
        % confusePair
        confusePairOffset = 4*(iter-(RESUME_FROM_SNAPSHOT_ITER+1));
        confusePair(1,confusePairOffset+1:confusePairOffset+4) = groundtruthLbl;
        confusePair(2,confusePairOffset+1:confusePairOffset+4) = lbl;
    end
    
    accTrain = mean(cell2mat(accs(RESUME_FROM_SNAPSHOT_ITER+1                                      : RESUME_FROM_SNAPSHOT_ITER+1548/MINIBATCHSIZE)));
    accVal   = mean(cell2mat(accs(RESUME_FROM_SNAPSHOT_ITER+1548/MINIBATCHSIZE+1                   : RESUME_FROM_SNAPSHOT_ITER+1548/MINIBATCHSIZE+200/MINIBATCHSIZE)));
    accTest  = mean(cell2mat(accs(RESUME_FROM_SNAPSHOT_ITER+1548/MINIBATCHSIZE+200/MINIBATCHSIZE+1 : RESUME_FROM_SNAPSHOT_ITER+1548/MINIBATCHSIZE+200/MINIBATCHSIZE+200/MINIBATCHSIZE)));
    assert(size(accs,2) == RESUME_FROM_SNAPSHOT_ITER+1548/MINIBATCHSIZE+200/MINIBATCHSIZE+200/MINIBATCHSIZE);
    fprintf('%d\t%f\t%f\t\t%f\n', RESUME_FROM_SNAPSHOT_ITER, accTrain, accVal, accTest);
    
    idx = 1:1548;
    M = confusionmat(confusePair(1,idx), confusePair(2,idx));
    R = round(diag(M)./sum(M,2) * 100);
    B = round(diag(M)'./sum(M,1) * 100);
    BR = round(sum(diag(M)) / sum(sum(M)) * 100);
    disp([M R
        B BR]);
    idx = 1548+1:1548+200;
    M = confusionmat(confusePair(1,idx), confusePair(2,idx));
    R = round(diag(M)./sum(M,2) * 100);
    B = round(diag(M)'./sum(M,1) * 100);
    BR = round(sum(diag(M)) / sum(sum(M)) * 100);
    disp([M R
        B BR]);
    idx = 1548+200+1:1548+200+200;
    M = confusionmat(confusePair(1,idx), confusePair(2,idx));
    R = round(diag(M)./sum(M,2) * 100);
    B = round(diag(M)'./sum(M,1) * 100);
    BR = round(sum(diag(M)) / sum(sum(M)) * 100);
    disp([M R
        B BR]);
end

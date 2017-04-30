function [loss] = checkff(TRAINX, TRAINY, MINIBATCHSIZE, c0P, c1P, b2P, f6P, b7P, d8mask, xO, c0O, t3O, s4O, flatO, b7O, d8O, smnO, yO, smlO)
    % x
    checkVal = TRAINX(:,:,:,1:MINIBATCHSIZE);% minibatch偏移
    if(~isempty(xO))
        fprintf('x\t%e\n', norm(reshape(checkVal-xO,[],1)));
    end
    
    % conv
    checkLast = checkVal;
    checkVal = zeros([32 24 5 MINIBATCHSIZE]);% 输出尺寸
    for i=1:MINIBATCHSIZE% N
        for j=1:5% 输出通道
            for k=1:3% 输入通道
                tmp = conv2(checkLast(:,:,k,i), c0P(:,:,k,j), 'same');
                checkVal(:,:,j,i) = checkVal(:,:,j,i) + tmp;
            end
        end
    end
    if(~isempty(c0O))
        fprintf('c\t%e\n', norm(reshape(checkVal-c0O,[],1)));
    end
    
    % conv bais tanh
    checkLast = checkVal;
    checkVal = zeros([32 24 10 MINIBATCHSIZE]);% 输出尺寸
    for i=1:MINIBATCHSIZE% N
        for j=1:10% 输出通道
            for k=1:5% 输入通道
                tmp = conv2(checkLast(:,:,k,i), c1P(:,:,k,j), 'same');
                checkVal(:,:,j,i) = checkVal(:,:,j,i) + tmp;
            end
            checkVal(:,:,j,i) = checkVal(:,:,j,i) + b2P(j);
        end
    end
    checkVal = tanh(checkVal);
    if(~isempty(t3O))
        fprintf('c\t%e\n', norm(reshape(checkVal-t3O,[],1)));
    end
    
    % maxpool
    sub1 = checkVal(1:2:end, 1:2:end, :, :);
    sub2 = checkVal(1:2:end, 2:2:end, :, :);
    sub3 = checkVal(2:2:end, 1:2:end, :, :);
    sub4 = checkVal(2:2:end, 2:2:end, :, :);
    sub1234 = cat(5, sub1, sub2, sub3, sub4);
    checkVal = max(sub1234, [], 5);
    if(~isempty(s4O))
        fprintf('s\t%e\n', norm(reshape(checkVal-s4O,[],1)));
    end
    
    % flatten
    checkVal = reshape(checkVal, [1 1 16*12*10 MINIBATCHSIZE]);
    if(~isempty(flatO))
        fprintf('flat\t%e\n', norm(reshape(checkVal-flatO,[],1)));
    end
    
    % fullcon bias tanh
    checkVal = permute(checkVal, [3 4 1 2]);
    checkVal = f6P * checkVal + repmat(b7P, [1 MINIBATCHSIZE]);
    checkVal = permute(checkVal, [3 4 1 2]);
    if(~isempty(b7O))
        fprintf('f\t%e\n', norm(reshape(checkVal-b7O,[],1)));
    end
    
    % dropout
    checkVal = checkVal .* d8mask;
    if(~isempty(d8O))
        fprintf('d\t%e\n', norm(reshape(checkVal-d8O,[],1)));
    end
    
    % smn
    checkVal = permute(checkVal, [3 4 1 2]);
    for i=1:MINIBATCHSIZE
        prob = checkVal(:,i);
        prob = exp(prob);
        prob = prob / sum(prob);
        checkVal(:,i) = prob;
    end
    checkVal = permute(checkVal, [3 4 1 2]);
    if(~isempty(smnO))
        fprintf('smn\t%e\n', norm(reshape(checkVal-smnO,[],1)));
    end
    
    % y
    checkVal2 = TRAINY(:,:,:,1:MINIBATCHSIZE);% minibatch偏移
    if(~isempty(yO))
        fprintf('y\t%e\n', norm(reshape(checkVal2-yO,[],1)));
    end
    
    % sml
    lbl = permute(checkVal2, [3 4 1 2]);
    [~, lbl] = max(lbl);
    idx = sub2ind([1 1 6 MINIBATCHSIZE], ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), lbl, 1:MINIBATCHSIZE);
    checkVal = checkVal(idx);
    checkVal = -log(checkVal);
    checkVal = mean(checkVal);
    if(~isempty(smlO))
        fprintf('sml\t%e\n', checkVal-smlO);
    end
    
    loss = checkVal;
end

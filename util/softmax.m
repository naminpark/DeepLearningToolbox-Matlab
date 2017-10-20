function y =softmax(x)
    
    [m,ndim] = size(x);
    if ndim >= 2
        x = x';
        x = x - max(x,[],1);
        y = exp(x)./ sum(exp(x),1);
        y=y';
        
        return
    end

    x= x- max(x);
    y = exp(x)./sum(exp(x));
end
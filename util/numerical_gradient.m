function grad= numerical_gradient(Func, w)

    h=1e-3;
    
    grad = zeros(size(w));
    
    for idx = 1:length(w)
        tmp_val = w(idx);
        % f(x+h)
        w(idx)= tmp_val+h;
        fxh1 = Func(w);
        % f(x-h)
        w(idx)= tmp_val-h;
        fxh2 = Func(w);
        
        grad(idx)=(fxh1 - fxh2) / (2*h);
        w(idx)=tmp_val;
    end
    
end


% numerical_gradient(@(x)sum(x.^2), [3,4])
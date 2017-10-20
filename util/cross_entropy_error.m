function out = cross_entropy_error(y,t)
    
    [m1,n1] = size(y);
    [m2,n2] = size(t);
    
    batch_size =m1;
    if n1 == 1
       y = reshape(y,1,m1);
       t = reshape(t,1,m1);
    end
    
    if m1 == m2
        [tmp,t] = max(t,[],2);
    end
    
    
    delta = 1e-7;
    
    out = -1*sum(log(diag(y(:,t))))/batch_size;
    
    %out = -1* sum(sum (t.*log(y+delta)))/batch_size;
end
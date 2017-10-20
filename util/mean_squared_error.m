function err= mean_squared_error(y,t)
    err =  0.5 * (sum((y-t)^2));
end
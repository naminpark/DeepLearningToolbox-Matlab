function y = numerical_diff(Func,x)
    h = 1e-4;
    y  = (Func(x+h) - Func(x-h))/(2*h);
end

%function call numerical_diff(@(x)X^2,5);
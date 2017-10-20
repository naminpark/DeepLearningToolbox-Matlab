function y= gradient_descent(Func, init_x, lr, step_num)

    if nargin < 3
        lr = 0.01;
        step_num =100;
    end
    x= init_x;
    for i = 1: step_num
        grad= numerical_gradient(Func,x)
        x =x-lr*grad;
    end
    y=x;
end

% gradient_descent(@(x)sum(x.^2),[-3,4])
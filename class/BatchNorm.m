classdef BatchNorm < handle
    
    properties
        
        gamma;
        beta;
        momentum;
        input_shape;
        
        running_mean;
        running_var;
        xn;
        batch_size;
        xc;
        std;
        dgamma;
        dbeta; 
        
        initial_flag;
        
    end
    
    methods
        function obj = BatchNorm(gamma,beta, ...
                momentum, running_mean,running_var)
            obj.gamma = gamma;
            obj.beta = beta;
            obj.momentum = momentum;
            obj.running_mean = running_mean;
            obj.running_var = running_var;
            obj.batch_size = 0;
            obj.xc = 0;
            obj.std = 0;
            obj.dgamma = 0;
            obj.dbeta = 0;
            obj.initial_flag=1;
            
        end
        
        function out = forward(obj, x, train_flag)
            
            obj.input_shape = size(x);
            
            if ndims(x) ~=2
                [N,C,H,W]= size(x);
                x = reshape(x',N,C*H*W);
                
                out = obj.forward_(x, train_flag);
            
                out = reshape(out,obj.input_shape)';
                return;
            end
            
            out = obj.forward_(x, train_flag);

            
        end
        
        function out = forward_(obj, x, train_flag)
            if obj.initial_flag == 1
                [N,D] = size(x);
                obj.running_mean = zeros(1,D);
                obj.running_var = zeros(1,D);
                obj.initial_flag = 0;
            end
            [N,D] = size(x);
            
            if train_flag == 1
                mu = mean(x,1);
                xc = x - mu;
                var = mean(x.^2, 1);
                std = sqrt(var + 10e-7);
                xn = xc ./ std;
                
                [obj.batch_size,gar] = size(x);
                obj.xc = xc;
                obj.xn = xn;
                obj.std =std;
                
                m = obj.momentum;
                
                obj.running_mean = m.* obj.running_mean + (1 - m) .* mu;
                obj.running_var = m.* obj.running_var + (1 - m) .* var;
                
            else
                xc = x - repmat(obj.running_mean,N,1);
                xn = xc ./ sqrt(obj.running_var+10e-7);
                
            end
            
            out = obj.gamma .* xn + repmat(obj.beta,N,1);
        end    
        
        function dx = backward(obj, dout)
            if length(size(dout)) ~= 2
                [N,C,H,W] = size(dout);
                dout = reshape(dout',N,C*H*W);
                dx = obj.backward_(dout);
                dx = reshape(dx,obj.input_shape)';
                return;
            end
            dx = obj.backward_(dout);
           
        end
        
        function dx = backward_(obj,dout)
            dbeta = sum(dout,1);
            dgamma = sum(obj.xn .* dout, 1);
            dxn = obj.gamma .* dout;
            dxc = dxn ./ obj.std;
            dstd = -1 * sum((dxn .*obj.xc) ./ (obj.std .*obj.std),1);
            
            dvar = 0.5 * dstd ./ obj.std;
            
            dxc = dxc + (2.0 / obj.batch_size) * obj.xc .* dvar;
            dmu = sum(dxc,1);
            dx =dxc - dmu ./ obj.batch_size;
            
            obj.dgamma =dgamma;
            obj.dbeta = dbeta;
        end
       
        function obj=update(obj,x,y)
            obj.gamma= x;
            obj.beta =y;
        end
        
    end
end
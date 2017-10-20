classdef MultiLayerNet < handle
   
    properties
        params;
        BN;
        layers;
        params_v;
        params_h;
        BN_v;
        BN_h;
        Drop;
        params_location =[];
        btNorm_location =[];
    end
    
    
    methods
        function obj = MultiLayerNet(Layers)
            
            params_idx =1;
            BN_idx =1;
            obj.params_location=zeros(1,numel(Layers));
            obj.btNorm_location=zeros(1,numel(Layers));
            for l = 1 : numel(Layers)   %  layer
                if strcmp(Layers{l}.type, 'Affine')
                    obj.params{params_idx}.W = 0.01 * randn(Layers{l}.in,Layers{l}.out);
                    obj.params{params_idx}.b = zeros(1,Layers{l}.out);
                    obj.params_v{params_idx}.W = 0;
                    obj.params_v{params_idx}.b = 0;
                    obj.params_h{params_idx}.W = 0;
                    obj.params_h{params_idx}.b = 0;
                    obj.layers{l}= Layers{l}.func(obj.params{params_idx}.W,obj.params{params_idx}.b);
                    params_idx=params_idx+1;
                    obj.params_location(l)=1;
                elseif strcmp(Layers{l}.type, 'Relu')  
                    obj.layers{l}= Layers{l}.func();
                    
                elseif strcmp(Layers{l}.type, 'BatchNorm')
                    
                    obj.BN{BN_idx}.gamma = ones(1,Layers{l}.size);  %%%%
                    obj.BN{BN_idx}.beta = zeros(1,Layers{l}.size);  %%%%
                    obj.BN_v{BN_idx}.gamma = 0;
                    obj.BN_v{BN_idx}.beta = 0;
                    obj.BN_h{BN_idx}.gamma = 0;
                    obj.BN_h{BN_idx}.beta = 0;
                    momentum = 0.9;
                    running_mean = 0;
                    running_var =0;
                    obj.layers{l}= Layers{l}.func(obj.BN{BN_idx}.gamma,obj.BN{BN_idx}.beta,...
                        momentum,running_mean,running_var);
                    BN_idx=BN_idx+1;
                    obj.btNorm_location(l)=1;
                
                elseif strcmp(Layers{l}.type, 'Dropout') 
                    obj.Drop = Layers{l}.dropout_ratio;
                    obj.layers{l}= Layers{l}.func(obj.Drop);
                    
                elseif strcmp(Layers{l}.type, 'SoftmaxWithLoss') 
                    obj.layers{l}= Layers{l}.func();
    
                end
            end
            
           
        end
        
        function [y] = predict(obj,x,flag)
            
            tmp=obj.layers{1}.forward(x,flag);
            for l = 2 : numel(obj.layers)-2   %  layer
                tmp=obj.layers{l}.forward(tmp,flag);
            end
            y=obj.layers{numel(obj.layers)-1}.forward(tmp,flag);
               
            %y=obj.layers.Last.forward(tmp);
        end
        
        function out = loss(obj, x,t)
            y= obj.predict(x,1);
            out =obj.layers{numel(obj.layers)}.forward(y,t);
        end  
        
        function out=accuracy(obj, x,t)
            y = obj.predict(x,0);
            [tmp, y] = max(y,[],2);
            
            [m,tdim]=size(t);
            
            if tdim ~=1
                [tmp, t] = max(t,[],2);
            end
            
            out = sum(y==t)/m;
           
        end
        
                
        function [params_grads,BN_grads] = gradient(obj,x,t)
           
           [out]=obj.loss(x,t); 
           dout = 1;
            
           dout=obj.layers{numel(obj.layers)}.backward(dout);
            
           for l = numel(obj.layers)-1 :-1: 1   %  layer
               dout=obj.layers{l}.backward(dout);
           end
            
           
           params_idx = 1;
           BN_idx = 1;
           for l = 1 : numel(obj.layers)   %  layer
                if obj.params_location(l) ==1
                    params_grads{params_idx}.W = obj.layers{l}.dW;
                    params_grads{params_idx}.b = obj.layers{l}.db;
                    params_idx=params_idx+1;
                elseif obj.btNorm_location(l) ==1
                    BN_grads{BN_idx}.gamma = obj.layers{l}.dgamma;
                    BN_grads{BN_idx}.beta = obj.layers{l}.dbeta;
                    BN_idx=BN_idx+1;
                end
           end     
           
       end

       function obj=SDG_update(obj,params_grads,BN_grads,lr)
           %obj.params.W1 = obj.params.W1 -lr*grad.W1;
           %obj.params.W2 = obj.params.W2 -lr*grad.W2;
           %obj.params.b1 = obj.params.b1 -lr*grad.b1;
           %obj.params.b2 = obj.params.b2 -lr*grad.b2;

           for i = 1 : numel(params_grads)   %  layer
                obj.params{i}.W = obj.params{i}.W - lr*params_grads{i}.W;
                obj.params{i}.b = obj.params{i}.b - lr*params_grads{i}.b;
           end 
           
           for i = 1 : numel(BN_grads)   %  layer
                obj.BN{i}.gamma = obj.BN{i}.gamma - lr*BN_grads{i}.gamma;
                obj.BN{i}.beta = obj.BN{i}.beta - lr*BN_grads{i}.beta;
           end 
           
           params_idx = 1;
           BN_idx = 1;
           for l = 1 : numel(obj.layers)   %  layer
                if obj.params_location(l) ==1
                    obj.layers{l}=obj.layers{l}.update...
                    (obj.params{params_idx}.W,  obj.params{params_idx}.b);
                    params_idx = params_idx+1;
                elseif obj.btNorm_location(l) ==1
                    obj.layers{l}=obj.layers{l}.update...
                    (obj.BN{idx}.gamma,  obj.BN{idx}.beta);
                    BN_idx = BN_idx+1;
                end
           end   
            
       end

       
       function obj=Momentum_update(obj,params_grads,BN_grads,lr)
           %obj.params.W1 = obj.params.W1 -lr*grads.W1;
           %obj.params.W2 = obj.params.W2 -lr*grads.W2;
           %obj.params.b1 = obj.params.b1 -lr*grads.b1;
           %obj.params.b2 = obj.params.b2 -lr*grads.b2;
           momentum = 0.9;
           
           for i = 1 : numel(params_grads)   %  layer
                obj.params_v{i}.W = momentum*obj.params_v{i}.W - lr*params_grads{i}.W;
                obj.params_v{i}.b = momentum*obj.params_v{i}.b - lr*params_grads{i}.b;
                
                obj.params{i}.W = obj.params{i}.W + obj.params_v{i}.W;
                obj.params{i}.b = obj.params{i}.b + obj.params_v{i}.b;
           end 
           
           for i = 1 : numel(BN_grads)   %  layer
                obj.BN_v{i}.gamma = momentum*obj.BN_v{i}.gamma - lr*BN_grads{i}.gamma;
                obj.BN_v{i}.beta = momentum*obj.BN_v{i}.beta - lr*BN_grads{i}.beta;
                
                obj.BN{i}.gamma = obj.BN{i}.gamma + obj.BN_v{i}.gamma;
                obj.BN{i}.beta = obj.BN{i}.beta + obj.BN_v{i}.beta;
           end 
 
           params_idx = 1;
           BN_idx = 1;
           for l = 1 : numel(obj.layers)   %  layer
                if obj.params_location(l) ==1
                    obj.layers{l}=obj.layers{l}.update...
                    (obj.params{params_idx}.W,  obj.params{params_idx}.b);
                    params_idx = params_idx+1;
                elseif obj.btNorm_location(l) ==1
                    obj.layers{l}=obj.layers{l}.update...
                    (obj.BN{BN_idx}.gamma,  obj.BN{BN_idx}.beta);
                    BN_idx = BN_idx+1;
                end
           end   
       end

       
       function obj=AdaGrad_update(obj,params_grads,BN_grads,lr)
           
           for i = 1 : numel(params_grads)   %  layer
                obj.params_h{i}.W = obj.params_h{i}.W + params_grads{i}.W.*params_grads{i}.W;
                obj.params_h{i}.b = obj.params_h{i}.b + params_grads{i}.b.*params_grads{i}.b;
                
                obj.params{i}.W = obj.params{i}.W - lr*(params_grads{i}.W) ./(sqrt(obj.params_h{i}.W)+ 1e-7);
                obj.params{i}.b = obj.params{i}.b - lr*(params_grads{i}.b) ./(sqrt(obj.params_h{i}.b)+ 1e-7);
                
           end 
           
           for i = 1 : numel(BN_grads)   %  layer
                obj.BN_h{i}.gamma = obj.BN_h{i}.gamma + BN_grads{i}.gamma.*BN_grads{i}.gamma;
                obj.BN_h{i}.beta = obj.BN_h{i}.beta + BN_grads{i}.beta.*BN_grads{i}.beta;
                
                obj.BN{i}.gamma = obj.BN{i}.gamma - lr*(BN_grads{i}.gamma) ./(sqrt(obj.BN_h{i}.gamma)+ 1e-7);
                obj.BN{i}.beta = obj.BN{i}.beta - lr*(BN_grads{i}.beta) ./(sqrt(obj.BN_h{i}.beta)+ 1e-7);
                
           end 
           
           params_idx = 1;
           BN_idx =1;
           for l = 1 : numel(obj.layers)   %  layer
                if obj.params_location(l) ==1
                    obj.layers{l}=obj.layers{l}.update...
                    (obj.params{params_idx}.W,obj.params{params_idx}.b);
                    params_idx = params_idx+1;
                elseif obj.btNorm_location(l) ==1
                    obj.layers{l}=obj.layers{l}.update...
                    (obj.BN{BN_idx}.gamma,obj.BN{BN_idx}.beta);
                    BN_idx = BN_idx+1;    
                end
           end 
       end

       
       
    end
    
    
end
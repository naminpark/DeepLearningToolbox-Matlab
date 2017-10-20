classdef SoftmaxWithLoss < handle
   
    properties
        loss;
        y;
        t;
    end
    
    
    methods
        function obj=SoftmaxWithLoss()
            
            obj.loss = 0;
            obj.y = 0;
            obj.t =0;
            
        end
        function [loss]=forward(obj,x,t)
            
            obj.t = t;
            obj.y = softmax(x);
            obj.loss = cross_entropy_error(obj.y, obj.t);
            loss = obj.loss;
            
        end
        
        function [dx]=backward(obj,dout)
           %dout = 1;
           batch_size = length(obj.t);
           dx = (obj.y-obj.t) / batch_size;
           
        end
    end
    
    
end
classdef simplenet < handle
   
    properties
        W = rand(2,3);
    end
    
    methods
        function out = predict(obj,x)
            out = x*obj.W;
        end
        
        function loss = loss(obj,x,t)
            z=obj.predict(x);
            y=softmax(z);
            loss = cross_entropy_error(y,t);
        end
    end
    
end
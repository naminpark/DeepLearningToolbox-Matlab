classdef Sigmoid < handle
   
    properties
        out =[]
    end
    
    methods
        function [out,obj]=forward(obj,x)
            out = 1./(1+exp(x));
            obj.out = out;
        end
        
        function [dx]=backward(obj,dout)
            dx = dout.*(1.0-obj.out).*obj.out;
        end
    end
    
end
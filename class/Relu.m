classdef Relu < handle

    properties
        mask =[];
    end
    
    
    methods
        function obj=Relu()
            obj.mask =0;
            
        end
        function [out]=forward(obj,x,flag)
            obj.mask = (x<=0);
            out=x;
            
            out(obj.mask)=0;
            
        end
        
        function [dx] = backward(obj, dout)
            dout(obj.mask)=0;
            dx = dout;
        end
    end
    
    
    
end


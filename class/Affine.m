classdef Affine < handle
   
    properties
        W;
        b;
        x;
        dW;
        db;
        
    end
    
    methods
        function obj=Affine(W,b)
            
            obj.W= W;
            obj.b= b;
            
        end
        
        
        function [out]=forward(obj,x,flag)
            obj.x=x;
            out =x*obj.W + obj.b;
        end
        
        function [dx]=backward(obj,dout)
            dx = dout*obj.W';
            obj.dW = obj.x'*dout;
            obj.db= sum(dout,1);
        end

        function obj = update(obj,W,b)
            obj.W = W;
            obj.b = b;
        end
        
    end
    
    
end
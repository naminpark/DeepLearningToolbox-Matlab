classdef Dropout < handle
   
    properties
        dropout_ratio;
        mask;
    end 
    
    methods
        function obj=Dropout(dropout_ratio)
            obj.dropout_ratio= dropout_ratio;
            obj.mask =[];
        end
        
        function [out]= forward(obj, x, train_flag)
            if train_flag == 1
                obj.mask = (rand(size(x)) > obj.dropout_ratio);
                out = x.* obj.mask;
                return;
            else
                out = x.* (1.0 - obj.dropout_ratio);
            end
        end
        
        function [out] = backward(obj, dout)
            out =dout.* obj.mask;
        end
        
    end
    
end
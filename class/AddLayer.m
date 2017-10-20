classdef AddLayer < handle

   properties
       
       x=0;
       y=0;
       
   end
   
   methods (Access = public)
       %function obj = MulLayer(obj)
       %    obj.x = 0;
       %    obj.y = 0;

       %end
       
       function [out] = forward(obj,w,z)
 
           
           out = w + z;
       end
       
       function [dx,dy] =backward(obj, dout)
           dx = dout * 1;
           dy = dout * 1;
       end
   end
end
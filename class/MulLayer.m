classdef MulLayer < handle

   properties
       
       x=0;
       y=0;
       
   end
   
   methods (Access = public)
       %function obj = MulLayer(obj)
       %    obj.x = 0;
       %    obj.y = 0;

       %end
       
       function [out,obj] = forward(obj,w,z)
           obj.x = w;
           obj.y = z;
           
           out = w * z;
       end
       
       function [dx,dy] =backward(obj, dout)
           dx = dout * obj.y;
           dy = dout * obj.x;
       end
   end
end
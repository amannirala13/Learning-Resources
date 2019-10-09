% SDPTA
clear all; clc;
c=1; E=0; n=5; P=6; k=1; w=randn(1,n+1);  
X = [-1 -4 -2 8 2 1 ; -1 -4 -2 9 2 1 ;  -1 -3 -1 8 10 4;
     -1 4 2 -8 -2 -1 ; -1 8 2 -9 -9 -9 ;  -1 10 11 -4 -2 -4 ];
d = [1 1 1 -1 -1 -1 ]; 
for i=1:1000
	E=0;
	for p=1:P 
      net=w*transpose(X(p,1:6));
      if net>0 o(p)=1; else o(p)=-1; end
      w=w+0.5*(d(1,p)-o(p))*X(p,1:6);
      E=E+1/2*(d(1,p)-o(p))^2;
      k=k+1;
   end
   	if E==0 
      break 
      else
      end
end
if E==0
  fprintf('Discrete Perceptron is trained and \n');
  fprintf('no of steps required for traing are = %d\n\n', k);
  fprintf('the weight vector is\n');
  fprintf('w = %7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n\n',w(1),w(2),w(3),w(4),w(5),w(6));
  fprintf('the output vector is equal to desired output vector\n');
  fprintf('o = %7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n\n',o(1),o(2),o(3),o(4),o(5),o(6));

else
  fprintf('Discrete Perceptron can not be trained in given steps\n');
  fprintf('may train if steps are increased\n');
  fprintf('w = %7.2f%7.2f%7.2f%7.2f%7.2f%7.2f\n\n\n\n',w(1),w(2),w(3),w(4),w(5),w(6));
end


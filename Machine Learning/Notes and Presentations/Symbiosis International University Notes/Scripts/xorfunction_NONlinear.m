%XOR function using MP-Model 
%1, -1,-1,1,1,1,1
%X1 X2 desired_ouput
%0 0    0
%0 1    1
%1 0    1
%1 1    0
clear;
clc;
%Getting weights and threshold value
disp('Enter weights');
w11=input('Weight w11=');
w12=input('weight w12=');
w21=input('Weight w21=');
w22=input('weight w22=');
v1=input('weight v1=');
v2=input('weight v2=');
disp('Enter Threshold Value');
theta=input('theta=');
x1=[0 0 1 1];
x2=[0 1 0 1];
z=[0 1 1 0];
con=1;
while con
zin1=x1*w11+x2*w21;
zin2=x1*w21+x2*w22;
for i=1:4
if zin1(i)>=theta
y1(i)=1;
else
y1(i)=0;
end
if zin2(i)>=theta
y2(i)=1;
else
y2(i)=0;
end
end
yin=y1*v1+y2*v2;
for i=1:4
if yin(i)>=theta;
y(i)=1;
else
y(i)=0;
end
end
disp('Output of Net');
disp(y);
if y==z
con=0;
else
disp('Net is not learning enter another set of weights and Threshold value');
w11=input('Weight w11=');
w12=input('weight w12=');
w21=input('Weight w21=');
w22=input('weight w22=');
v1=input('weight v1=');
v2=input('weight v2=');
theta=input('theta=');
end
end
disp('McCulloch-Pitts Net for XOR function');
disp('Weights of Neuron Z1');
disp(w11);
disp(w21);
disp('weights of Neuron Z2');
disp(w12);
disp(w22);
disp('weights of Neuron Y');
disp(v1);
disp(v2);
disp('Threshold value');
disp(theta);
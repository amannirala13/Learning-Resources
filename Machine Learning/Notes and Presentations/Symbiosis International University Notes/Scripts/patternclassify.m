clear
p = 5; % dimensionality of the augmented input space
N = 50; % number of training patterns - size of the training epoch
% PART 1: Generation of the training and validation sets.
X = 2*rand(p-1, 2*N)-1;
nn = round((2*N-1)*rand(N,1))+1;
X(:,nn) = sin(X(:,nn));
X = [X; ones(1,2*N)];
wht = 3*rand(1,p)-1; wht = wht/norm(wht);
wht
D = (wht*X >= 0);
Xv = X(:, N+1:2*N) ;
Dv = D(:, N+1:2*N) ;
X = X(:, 1:N) ;
D = D(:, 1:N) ;
% [X; D]
pr = [1, 3];
Xp = X(pr, :);
wp = wht([pr p]); % projection of the weight vector
c0 = find(D==0); c1 = find(D==1);
% c0 and c1 are vectors of pointers to input patterns X
% belonging to the class 0 or 1, respectively.
figure(1), clf reset
plot(Xp(1,c0),Xp(2,c0),'o', Xp(1, c1), Xp(2, c1),'x')
% The input patterns are plotted on the selected projection
% plane. Patterns belonging to the class 0, or 1 are marked
% with 'o' , or 'x' , respectively
axis(axis), hold on
% The axes and the contents of the current plot are frozen
% Superimposition of the projection of the separation plane on the
% plot. The projection is a straight line. Four points lying on this
% line are found from the line equation wp . x = 0
L = [-1 1] ;
S = -diag([1 1]./wp(1:2))*(wp([2,1])'*L +wp(3)) ;
plot([S(1,:) L], [L S(2,:)]), grid, drawnow
% PART 2: Learning
eta = 0.5; % The training gain.
wh = 2*rand(1,p)-1;
% Random initialisation of the weight vector with values
% from the range [-1, +1]. An example of an initial
% weight vector follows
% Projection of the initial decision plane which is orthogonal
% to wh is plotted as previously:
wp = wh([pr p]); % projection of the weight vector
S = -diag([1 1]./wp(1:2))*(wp([2,1])'*L +wp(3)) ;
plot([S(1,:) L], [L S(2,:)]), grid on, drawnow
C = 50; % Maximum number of training epochs
E = [C+1, zeros(1,C)]; % Initialization of the vector of the total sums of squared errors over an epoch.
WW = zeros(C*N, p); % The matrix WW will store all weight
% vector whone weight vector per row of the matrix WW
c = 1; % c is an epoch counter
cw = 0 ; % cw total counter of weight updates
while (E(c)>1)|(c==1)
c = c+1;
plot([S(1,:) L], [L S(2,:)], 'w'), drawnow
for n = 1:N
eps = D(n) - ((wh*X(:,n)) >= 0); % eps(n) = d(n) - y(n)
wh = wh + eta*eps*X(:,n)'; % The Perceptron Learning Law
cw = cw + 1;
WW(cw, :) = wh/norm(wh); % The updated and normalised weight vector is stored in WW for feature plotting
E(c) = E(c) + abs(eps) ; % |eps| = eps^2
end;
wp = wh([pr p]); % projection of the weight vector
S = -diag([1 1]./wp(1:2))*(wp([2,1])'*L +wp(3)) ;
plot([S(1,:) L], [L S(2,:)], 'g'), drawnow
end;
% After every pass through the set of training patterns the projection of the current decision plane which is determined by the current weight vector is plotted after the previous projection has been erased.
WW = WW(1:cw, pr);
E = E(2:c+1)
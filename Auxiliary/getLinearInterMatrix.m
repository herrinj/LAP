function A= getLinearInterpolationMatrix(omega,m,x)
% =========================================================================
% (c) Lars Ruthotto 2015/12/03
% http://www.mathcs.emory.edu/~lruthot/
%
% function A = getLinearInterpolationMatrix(omega,m,x)
%
% builds interpolation matrix A, s.t. A(x)*I(:) = linearInter(I,omega,x)
%
%
% Input:
%  omega - description of computational domain
%  m     - discretization size of image
%  x     - interpolation points
%
% Output:
%  A     - linear interpolation matrix
%
% see also linearInter
% =========================================================================

if nargin==0,
    help(mfilename);
    runMinimalExample;
    return;
end
h   = (omega(2:2:end)-omega(1:2:end))./m;
dim = length(omega)/2;
x   = reshape(x,[],dim);
n   = size(x,1);

% Convert x and y to the coordinate system 1:m, 1:n
for i=1:dim,
    x(:,i) = (x(:,i)-omega(2*i-1))/h(i) + 1/2;
end

Valid = @(j) (0<x(:,j) & x(:,j)<m(j)+1);      % determine indices of valid points

switch dim,
    case 1, valid = find( Valid(1) );
    case 2, valid = find( Valid(1) & Valid(2) );
    case 3, valid = find( Valid(1) & Valid(2) & Valid(3) );
    otherwise, error('%s - dimension must be 1,2 or 3',mfilename);
end;

if isempty(valid),
    A = sparse(n,prod(m)); return;
end;

P = floor(x); x = x-P;                        % split x into integer/remainder
p = @(j) P(valid,j); xi = @(j) x(valid,j);

switch dim
    case 1
        ij   = [       valid,p(1)  ,(1-xi(1))];
        ij   = [ij;    valid,p(1)+1,xi(1)];
        
        % delete rows where particles run out of the domain
        valid = (ij(:,2)>0).*(ij(:,2)<=m(1))==1;
        ij = ij(valid,:);
        
        A = sparse(ij(:,1),ij(:,2),ij(:,3),n,prod(m));
    case 2
        ind  = valid;
        
        % get indices of neighbouring points
        jk    = [p(1)  ,p(2)  ];
        j1k   = [p(1)+1,p(2)  ];
        jk1   = [p(1)  ,p(2)+1];
        j1k1  = [p(1)+1,p(2)+1];
        
        % compute weights for interpolation
        ss    = [(1-xi(1)).*(1-xi(2)) ; (xi(1))  .*(1-xi(2)); ...
                 (1-xi(1)).*(xi(2))   ; (xi(1))  .*(xi(2)) ];
        
        ii = [ind; ind; ind; ind ];
        jj = [jk ; j1k; jk1; j1k1];
        
        % points that are outise the domain
        valid =    (jj(:,1)>0).*(jj(:,1)<=m(1))...
                 .*(jj(:,2)>0).*(jj(:,2)<=m(2))==1;
      
        A = sparse(ii(valid),jj(valid,1)+(jj(valid,2)-1)*m(1),ss(valid),n,prod(m));
    case 3
        ij   = [    valid,p(1)  ,p(2)  ,p(3)  , (1-xi(1)).*(1-xi(2)).*(1-xi(3)) ];
        ij   = [ij; valid,p(1)+1,p(2)  ,p(3)  , (xi(1))  .*(1-xi(2)).*(1-xi(3)) ];
        ij   = [ij; valid,p(1)  ,p(2)+1,p(3)  , (1-xi(1)).*(xi(2))  .*(1-xi(3)) ];
        ij   = [ij; valid,p(1)+1,p(2)+1,p(3)  , (xi(1))  .*(xi(2))  .*(1-xi(3)) ];
        ij   = [ij; valid,p(1)  ,p(2)  ,p(3)+1, (1-xi(1)).*(1-xi(2)).*xi(3) ];
        ij   = [ij; valid,p(1)+1,p(2)  ,p(3)+1, (xi(1))  .*(1-xi(2)).*xi(3) ];
        ij   = [ij; valid,p(1)  ,p(2)+1,p(3)+1, (1-xi(1)).*(xi(2))  .*xi(3) ];
        ij   = [ij; valid,p(1)+1,p(2)+1,p(3)+1, (xi(1))  .*(xi(2))  .*xi(3) ];
        
        % delete rows where particles run out of the domain
        valid =    (ij(:,2)>0).*(ij(:,2)<=m(1))...
            .*(ij(:,3)>0).*(ij(:,3)<=m(2))...
            .*(ij(:,4)>0).*(ij(:,4)<=m(3))==1;
        ij = ij(valid,:);
        
        
        A = sparse(ij(:,1),ij(:,2)+(ij(:,3)-1)*m(1)+(ij(:,4)-1)*m(1)*m(2),ij(:,5),n,prod(m));
    case 3
        error('nyi')
    otherwise
        error('%s - dimension must be 1,2 or 3',mfilename);
end



function runMinimalExample

% 1D example
omega = [0,10];
Tdata = [0,1,4,1,0];
Tcoef = Tdata;
m     = length(Tdata);
xc    = linspace(-4,11,101);
Ty    = linearInter(Tcoef,omega,xc);

A     = feval(mfilename,omega,m,xc(:));

err = Ty(:) - A*Tdata(:);
fprintf('1D - error: %e \n',norm(err(:)))


% 2D example
omega = [0,10,0,8];
Tdata = [1,2,3,4;1,2,3,4;4,4,4,4]; m = size(Tdata);
Tcoef = Tdata;
xc    = getCellCenteredGrid(omega+[-1 1 -1 1],5*m);

A = feval(mfilename,omega,m,xc);

Ty = linearInter(Tcoef,omega,xc);

err = (Ty(:) - A*Tcoef(:));
fprintf('2D - error: %e \n',norm(err(:)))


% 3D example
omega = [0,1,0,2,0,1]; m = [13,16,7];
Xdata = getCellCenteredGrid(omega,m);
Y     = reshape(Xdata,[m,3]);
Tdata = (Y(:,:,:,1)-0.5).^2 + (Y(:,:,:,2)-0.75).^2 + (Y(:,:,:,3)-0.5).^2 <= 0.15;
Tcoef = reshape(Tdata,m);
xc    = getCellCenteredGrid(omega+randn(1,6),4*m);

A = feval(mfilename,omega,m,xc);

% test if T(y) = A(y)' * rho
Ty = linearInter(Tcoef,omega,xc);

err = (Ty(:) - A*Tcoef(:));
fprintf('3D - error: %e \n',norm(err(:)))




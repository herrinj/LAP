% =========================================================================
% (c) Lars Ruthotto 2015/12/03
% http://www.mathcs.emory.edu/~lruthot/
%
% function [Jc,para,dJ,H] = VarproSRObjFctn(wc,d, omega,mf, mc, varargin)
%
% VarPro (reduced) objective function for PIR super-resolution problem.
%
% J[W] = .5 sum_i^nv || A[y(w_i)]*x[w] - d_i  ||^2 + alpha ||L*x ||^2
%
% where A denotes an interpolation matrix, y(w_i) is a parametric
% transformation, and x is a high resolution reconstruction based on
% low-resolution data d_i. 
%
% Input:
%     wc  - current transformation parameters
%      d  - low-resolution data
%  omega  - description of computational domain
%     mf  - spatial discretization of high resolution image
%     mc  - spatial discretization of coarse resolution image        
%
% Output:
%   Jc    - function value
%   para  - parameter for plots
%   dJ    - gradient
%   H     - GaussNewton style approximation to Hessian (either explicit or as operator)
%
% see also
% =========================================================================

function [Jc,para,dJ,H] = VarproSRObjFctn(wc, d, omega, mf, mc, varargin)

if nargin==0
   % Load data
   setup2DSuperResProb;
   
   fctn = @(w) VarproSRObjFctn(w, d, omega, mf, mc);
   n = prod(size(w0(:,2:end)));
   wc = randn(n,1);
   
   % Test derivative
   n    = 10;
   h    = logspace(-1,-10,n);
   pert = randn(size(wc));

   T0 = zeros(n,1);
   T1 = zeros(n,1);
   T2 = zeros(n,1);

   [Jc,~,dJ,H] = fctn(wc);

    fprintf('h \t\t |f(x+hs) - f(x)| \t |f(x+hs) - f(x) - h*s''*df(x)|\n');
    for j = 1:n
        Jpert = fctn(wc + h(j)*pert);
        T0(j) = abs(Jpert - Jc);
        T1(j) = abs(Jpert - Jc - h(j)*pert'*dJ(:));
        T2(j) = abs(Jpert - Jc - h(j)*pert'*dJ(:) - 0.5*h(j)*h(j)*pert'*H*pert);

        fprintf('%1.4e \t %1.4e \t\t %1.4e \t\t %1.4e \n', h(j),T0(j),T1(j),T2(j));
    end

    figure();
   loglog(h,T0); hold on;
   loglog(h,T1); 
   loglog(h,T2); 
   legend('T0','T1','T2');
   
   para  = [];
   return;
end

matrixFree      = 0;   
regularizer     = 'grad';
doKaufman       = false;
solverLS        = 'bidiag';
iters           = 20;
alpha           = 0;

for k=1:2:length(varargin)     % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

doDerivative = nargout>2;
md           = size(d);
nVol         = md(end);
hc           = (omega(2:2:end)-omega(1:2:end))./mc;
hf           = (omega(2:2:end)-omega(1:2:end))./mf;

% Add column of zeros for first frame
wc           = reshape([trafo('w0'); wc(:)],[],nVol);

% Build interpolation matrix, i.e. A = K*T(w) with K downsampling, T(w) motion
grid    = getCellCenteredGrid(omega,mc);
Ai      = cell(nVol,1);
for i=1:nVol
    [yi{i}, dy{i}] = trafo(wc(:,i),grid); % Transform points
    Ai{i}       = getLinearInterMatrix(omega, mf, yi{i}); % Interpolation matrices
end

% Set up the regularizer
switch regularizer
    case{'grad'}
        S  = sqrt(prod(hf))*getGradient(omega,mf);
    case{'eye'}
        S = sqrt(prod(hf))*speye(prod(mf));
end

% Set up the linear problem in the image
if not(matrixFree)
    A       = cat(1,Ai{:});
    op      = [sqrt(prod(hc))*A;    sqrt(alpha)*S]; 
    rhs     = [sqrt(prod(hc))*d(:); zeros(size(S,1),1)];
    [m,n]   = size(op);
else
    A       = @(x,flag) sqrt(prod(hc))*opA(x,Ai,flag);
    op      = @(x,flag) opLS(x,A,S,alpha,prod(md),flag);
    rhs     = [sqrt(prod(hc))*d(:); zeros(size(S,1),1)];
    m       = numel(rhs);
    n       = prod(mf);
end

% Solve least squares problem for the current image
switch solverLS
    case{'bidiag'}
        [U,B,V] = lancBiDiag(op,rhs,iters,m,n); % No reorthogonalization 
        [Ub,Sb,Vb] = svd(full(B));
        U = U*Ub; U = U(:,1:iters); 
        Sb = Sb(1:iters,1:iters);
        V = V*Vb;
        xc = V*(Sb\(U'*rhs));
    case{'lsqr'}
        [xc,~,~,iters] = lsqr(op,rhs,1e-8,iters);
end
 
% Compute distance term
if not(matrixFree)
    Rc = A*xc;
else
    Rc = opA(xc,Ai,'notransp');
end

% Evaluate objective function
res     = sqrt(prod(hc))*(Rc - d(:));
Dc      = 0.5*(res'*res);
Sx      = S*xc(:);
Sc      = 0.5*alpha*(Sx)'*(Sx);
Jc      = Dc + Sc;

% Parameters for outside visualization
para = struct('Jc',Jc,'Dc',Dc,'Sc',Sc,'omega',omega,'m',mf,'mc',mc,'Tc',xc,'Rc',Rc,'nVol',nVol);

if not(doDerivative), return; end

res = reshape(res,[],nVol);
dJ  = []; H = cell(nVol-1,1);
xc = reshape(xc,mf);
Ur = U(1:numel(d),:);
Ur = reshape(Ur,[prod(mc) nVol iters]);

for i=2:nVol
    [~,dx] = linearInter(xc, omega, yi{i});
    Dkc = sqrt(prod(hc))*dx*dy{i};
    Ui  = squeeze(Ur(:,i,:));
    dJi = Dkc - Ui*(Ui'*Dkc);
    
    if not(doKaufman)
        % compute B part
        [~,dBi] = getPICMatrixAnalyticIntegral(omega,mf,mf,yi{i});
        DTr = sqrt(prod(hc))*dBi(res(:,i))*dy{i};
        dJi  =  dJi + Ui*(Sb'\(V'*DTr));
    end
    dJ = [dJ,res(:,i)'*dJi];
    H{i-1} = dJi'*dJi;
end

H = blkdiag(H{:});

end

function [out_vec] = opA(in_vec, Ai, flag)
%
%   [out_vec] = getA(in_vec, A, flag)
%
%   Computes A*in_vec or A'*in_vec for cell array Ai representing a block
%   rectangular matrix A = cat(1,Ai{:})
%
%   Input:   in_vec - input matrix of appropriate size for multiplication
%                Ai - cell array of matrices
%              flag - 'transp' or 'notransp' flag
%
%   Output: out_vec - output vector A*in_vec or A'*in_vec
%

nVol = length(Ai);
switch flag
    case{'notransp'}
        out_vec = zeros(size(Ai{1},1),nVol);
        for j = 1:nVol
            out_vec(:,j) = Ai{j}*in_vec;
        end
        out_vec = out_vec(:);
    case{'transp'}
        in_vec = reshape(in_vec,[],nVol);
        out_vec = zeros(size(Ai{1},2),1);
        for j = 1:nVol
           out_vec = out_vec + Ai{j}'*in_vec(:,j);
        end
end

end

function out_vec = opLS(in_vec,A,S,alpha,md,flag)
    switch flag
        case{'notransp'}
            out_vec = [A(in_vec,'notransp');sqrt(alpha)*(S*in_vec)];
        case{'transp'}
            in1 = in_vec(1:md);
            in2 = in_vec(md+1:end);
            out_vec = A(in1,'transp') + sqrt(alpha)*(S'*in2);
    end
end


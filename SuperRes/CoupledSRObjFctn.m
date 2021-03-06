% =========================================================================
%
% function [Jc,para,dJ,J] = CoupledSRObjFctn(yc, d,omega, mf, mc, varargin)
%
% Fully coupled objective function for PIR super-resolution problem.
%
% J[T,W] = .5 sum_i^nv || A[y(w_i)]*x - d_i  ||^2 + alpha ||S*x ||^2
%
% where A denotes an interpolation matrix, y(W_i) is a parametric
% transformation, and x is a high resolution reconstruction based on
% low-resolution data d_i. The regularization matrix S is a discrete
% gradient or identity matrix.
%
% Input:
%     yc  - current iterate, column vector with two components, i.e. yc = [xc(:); wc(:)]; 
%      d  - low-resolution data
%  omega  - description of computational domain
%     mf  - spatial discretization of high resolution image
%     mc  - spatial discretization of coarse resolution image
%                       
%
% Output:
%   Jc    - function value
%   para  - parameter for plots
%   dJ    - gradient
%   J     - Jacobian structure to approximate Hessian
%
% see also
% =========================================================================
function [Jc,para,dJ,J] = CoupledSRObjFctn(yc, d, omega, mf, mc, varargin)

if nargin==0
    % Load data
    setup2DSuperResProb;   
    
   fctn = @(y) CoupledSRObjFctn(y, d, omega, mf, mc);
   n = prod(mf) + prod(size(w0(:,2:end)));
   yc = randn(n,1);
   
   % Test derivative
   checkDerivative(fctn,yc);
   
   Jc   = [];
   para  = [];
   return;
end

matrixFree  = 1;
alpha       = 0;
regularizer = 'grad';

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

doDerivative = nargout>2;
md           = size(d);
nVol         = md(end);
hc           = (omega(2:2:end)-omega(1:2:end))./mc;
hf           = (omega(2:2:end)-omega(1:2:end))./mf;

% Split yc into the image xc and transformation parameters wc = [w1,..,wn];
xc = yc(1:prod(mf));
wc = reshape([trafo('w0');yc(prod(mf)+1:end)],[],nVol); % First frame fixed

% Build interpolation matrix, i.e. A = K*T(w) with K downsampling, T(w) motion
grid    = getCellCenteredGrid(omega,mc);
Ai      = cell(nVol,1);
dy      = cell(nVol,1);
dx      = cell(nVol,1);
for i=1:nVol
    [yi, dy{i}] = trafo(wc(:,i),grid); % Transform points
    Ai{i}       = getLinearInterMatrix(omega, mf, yi); % Interpolation matrices
    if doDerivative
        [~, dx{i}] = linearInter(reshape(xc,mf), omega, yi); % Derivatives of interpolation
    end
end

% Construct block-rectangular operator
if not(matrixFree)
    A   = cat(1,Ai{:});
    Rc  = A*xc(:);
else 
    Rc  = opA(xc,Ai,'notransp');
end

% Evaluate objective function
res     = sqrt(prod(hc))*(Rc - d(:));
Dc      = 0.5*(res'*res);

% Evaluate regularizer
switch regularizer 
    case{'grad'}
        S   = sqrt(prod(hf))*getGradient(omega,mf);
        Sx  = S*xc(:);
        Sc  = 0.5*alpha*(Sx'*Sx); 
    case{'eye'}
        S   = sqrt(prod(hf))*speye(prod(mf));
        Sx  = S*xc(:);
        Sc  = 0.5*alpha*(Sx'*Sx);
    case{'hybr'}
        S   = 0;
        Sx  = 0.0;
        Sc  = 0.0;
end

% Combine data misfit and regularizer
Jc  = Dc+Sc;

% Parameters for outside visualization
para = struct('Jc',Jc,'Dc',Dc,'Sc',Sc,'omega',omega,'m',mf,'mc',mc,'Tc',xc,'Rc',Rc,'nVol',nVol);


if not(doDerivative), return; end

% Jacobian w.r.t transformation w
Jw = sqrt(prod(hc))*sparse(blkdiag(dx{:})*blkdiag(dy{:}));
Jw = Jw(:,length(trafo('w0'))+1:end); % First frame fixed

% Jacobian w.r.t. image and gradient
if not(matrixFree)
    Jx = sqrt(prod(hc))*A;
    dJ = [res'*Jx + alpha*(Sx'*S), res'*Jw];
else
    Jx = @(x,flag) sqrt(prod(hc))*opA(x,Ai,flag);
    dJ = [Jx(res,'transp')' + alpha*(Sx'*S), res'*Jw];
end
        
% Load everything into Jacobian structure
J = struct('Jx', Jx, 'Jw', Jw, 'res', res, 'yc',yc, 'alpha', alpha, 'S', S, 'xdim', prod(mf), 'wdim', length(yc(prod(mf)+1:end)));

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

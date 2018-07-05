% =========================================================================
%
% function [Jc,para,dJ,J] = CoupledMRIObjFctn(yc, d, A, C, omega, m, varargin)
%
% Fully coupled objective function for MRI motion correction problem
%
% J[yc] = .5 sum_i^nv || A_i*F*C*T[w_i]*x - d_i ||^2 + alpha ||L*x||^2
%
% where A_i denotes a sampling pattern, F is a two dimensional Fourier 
% transformation, C is a block rectangular matrix of coil sensitivies, 
% T(y(W_i)_ is a parametric transformation, and x is a high resolution 
% reconstruction based on low-resolution data d_i. The regularization 
% matrix L is a discrete gradient or identity matrix.
%
% Input:
%     yc  - current iterate, column vector with two components, i.e. yc = [xc(:); wc(:)]; 
%      d  - Fourier sampling data
%      A  - cell array with sampling patters for i = 1:nv samples
%      C  - 3 dimensional array of coil sensitivities 
%  omega  - description of computational domain
%      m  - spatial discretization of image
%                       
%
% Output:
%   Jc    - function value
%   para  - parameter for plots
%   dJ    - gradient
%   J     - Jacobian structure to approximate Hessian
%
% =========================================================================

function [Jc,para,dJ,J] = CoupledMRIObjFctn(yc, d, A, C, omega, m, varargin)

if nargin==0
    % Load data
    setupMoCoMRIProb;   
    
   fctn = @(y) CoupledMRIObjFctn(y, d, A, C, omega, m,'alpha',0.0);
   m = prod(m);
   p = numel(w0(:,2:end));
   yc = [x0(:); reshape(w0(:,2:end),[],1)];
   pert = [randn(m,1) + 1i*randn(m,1); zeros(p,1)];
   
   % Test derivative
   n    = 10;
   h    = logspace(-1,-10,n);

   T0 = zeros(n,1);
   T1 = zeros(n,1);
   T2 = ones(n,1);

   [Jc,~,dJ,~] = fctn(yc);

   fprintf('h \t\t |f(x+hs) - f(x)| \t |f(x+hs) - f(x) - h*s''*df(x)|\n');
   for j = 1:n
       Jpert = fctn(yc + h(j)*pert);
       T0(j) = abs(Jpert - Jc);
       T1(j) = abs(Jpert - Jc - h(j)*real(pert.'*dJ(:)));
       %T2(j) = abs(Jpert - Jc - h(j)*pert'*dJ(:) - 0.5*h(j)*h(j)*pert'*H*pert);

       fprintf('%1.4e \t %1.4e \t\t %1.4e \t\t %1.4e \n', h(j),T0(j),T1(j),T2(j));
   end

   figure();
   loglog(h,T0); hold on;
   loglog(h,T1); 
   loglog(h,T2); 
   legend('T0','T1','T2');
   
   Jc   = [];
   para  = [];
   return;
end

alpha       = 0.0;
regularizer = 'grad';

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

doDerivative = nargout>2;
nSamples     = length(A);
nCoils       = size(C,3);

% Split yc into the image xc and transformation parameters wc = [w1,..,wn];
xc = yc(1:prod(m));
wc = reshape([trafo('w0');yc(prod(m)+1:end)],[],nSamples); % First frame fixed

% Build interpolation matrices T(w)
y       = cell(nSamples,1);
dy      = cell(nSamples,1);
Tw      = cell(nSamples,1);
dTx     = cell(nSamples,1);
grid    = getCellCenteredGrid(omega,m);
for k=1:nSamples
    [y{k},dy{k}] = trafo(wc(:,k),grid); % Transform points
    Tw{k} = getLinearInterMatrix(omega,m,trafo(wc(:,k),y{k})); % Interpolation matrices
    
    if doDerivative 
        [~,dTx_real] = linearInter(reshape(real(xc),m),omega,y{k});
        [~,dTx_imag] = linearInter(reshape(imag(xc),m),omega,y{k});
        dTx{k} = dTx_real*dy{k} + 1i*dTx_imag*dy{k}; % Combine real and imaginary parts
    end
end


% Evaluate objective function
Rc  = getAFCT(xc,A,C,Tw,m,'notransp');
res     = Rc - d(:);
Dc      = 0.5*(res'*res);

% Evaluate regularizer
switch regularizer 
    case{'grad'}
        S   = getGradient(omega,m);
        Sx  = S*xc(:);
        Sc  = 0.5*alpha*(Sx'*Sx); 
    case{'eye'}
        S   = speye(prod(m));
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
para = struct('Jc',Jc,'Dc',Dc,'Sc',Sc,'omega',omega,'m',m,'Tc',xc,'Rc',Rc,'nSamples',nSamples, 'nCoils', nCoils);


if not(doDerivative), return; end

% Jacobian w.r.t complex image x
Jx = @(x,flag) getAFCT(x,A,C,Tw,m,flag);

% Jacobian w.r.t transformation w
Jw = cell(nSamples,1);
for k = 2:nSamples
     Jw{k}(:,1) = reshape(A{k}*reshape(fft2(C.*reshape(dTx{k}(:,1),m)),[],nCoils),[],1);
     Jw{k}(:,2) = reshape(A{k}*reshape(fft2(C.*reshape(dTx{k}(:,2),m)),[],nCoils),[],1);
     Jw{k}(:,3) = reshape(A{k}*reshape(fft2(C.*reshape(dTx{k}(:,3),m)),[],nCoils),[],1);
end
Jw{1} = zeros(size(A{1},1)*nCoils,3);
Jw = sparse(blkdiag(Jw{:}));
Jw = Jw(:,4:end)/sqrt(prod(m));

% Gradient
dJ = [Jx(res,'transp')' + alpha*(Sx'*S), real(res'*Jw)];

% Load everything into Jacobian structure
J = struct('Jx', Jx, 'Jw', Jw, 'res', res, 'yc', yc, 'alpha', alpha, 'S', S, 'xdim', prod(m), 'wdim', length(yc(prod(m)+1:end)));

end


% =========================================================================
%
% function [Jc,para,dJ,J] = VarproMRIObjFctn(wc, d, A, C, omega, m, varargin)
%
% VarPro (reduced) objective function for MRI Motion Correction problem.
%
% J[yc] = .5 sum_i^nv || A_i*F*C*T[w_i]*x - d_i ||^2 + alpha ||L*x||^2
%
% where A_i denotes a sampling pattern, F is a two dimensional Fourier 
% transformation, C is a block rectangular matrix of coil sensitivies, 
% T(y(W_i)_ is a parametric transformation, and x is a high resolution 
% reconstruction based on Fourier sampled data d_i. The regularization 
% matrix L is a discrete gradient or identity matrix.
%
% Input:
%     wc  - current transformation parameters
%      d  - Fourier sampling data
%      A  - cell array with sampling patters for i = 1:nv samples
%      C  - 3 dimensional array of coil sensitivities 
%  omega  - description of computational domain
%      m  - spatial discretization of image
%                       
%
% Output:
%   Jc    - function value
%   para  - parameters for plots
%   dJ    - gradient
%   H     - GaussNewton style approximation to Hessian (either explicit or as operator)
%
% =========================================================================

function [Jc,para,dJ,H] = VarproMRIObjFctn(wc, d, A, C, omega, m, varargin)

if nargin==0
   % Load data
   setupMoCoMRIProb;   
    
   fctn = @(w) VarproMRIObjFctn(w, d, A, C, omega, m,'alpha',0.01,'regularizer','eye');
   p = numel(w0(:,2:end));
   wc = reshape(w0(:,2:end),[],1);
   pert = randn(p,1);
   
   % Test derivative
   n    = 10;
   h    = logspace(-1,-10,n);

   T0 = zeros(n,1);
   T1 = zeros(n,1);
   T2 = ones(n,1);

   [Jc,~,dJ,H] = fctn(wc);

   fprintf('h \t\t |f(x+hs) - f(x)| \t |f(x+hs) - f(x) - h*s''*df(x)|\n');
   for j = 1:n
       Jpert = fctn(wc + h(j)*pert);
       T0(j) = abs(Jpert - Jc);
       T1(j) = abs(Jpert - Jc - h(j)*real(pert.'*dJ(:)));
       T2(j) = abs(Jpert - Jc - h(j)*pert'*dJ(:) - 0.5*h(j)*h(j)*pert'*H*pert);

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

alpha       = 0.00;
regularizer = 'grad';
iters       = 100;

for k=1:2:length(varargin)     % overwrites default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

doDerivative = nargout>2;
nSamples     = length(A);
sampleSize   = size(A{1},1); % Assumes all samples same size
nCoils       = size(C,3);

% Add column of zeros for first frame
wc      = reshape([trafo('w0'); wc(:)],[],nSamples);

% Build interpolation matrices T(w)
y       = cell(nSamples,1);
dy      = cell(nSamples,1);
Tw      = cell(nSamples,1);
dTx     = cell(nSamples,1);
grid    = getCellCenteredGrid(omega,m);
for k=1:nSamples
    [y{k},dy{k}] = trafo(wc(:,k),grid); % Transform points
    Tw{k} = getLinearInterMatrix(omega,m,trafo(wc(:,k),y{k})); % Interpolation matricese
end     

% Set up regularizer
switch regularizer 
    case{'grad'}
        S   = getGradient(omega,m); 
    case{'eye'}
        S   = speye(prod(m));
end
                                                                                           
% Solve linear least squares problem for the image                                                               
rhs     = [d(:); zeros(size(S,1),1)];
AFCT    = @(x,flag) getAFCT(x(:),A,C,Tw,m,flag);
op      = @(x,flag) opLS(x, AFCT, S, alpha, nSamples*nCoils*sampleSize, flag);
[xc,~,~,iters] = lsqr(op,rhs,1e-8,iters);

% Evaluate objective function
Rc      = AFCT(xc,'notransp');
res     = Rc - d(:);
Dc      = 0.5*(res'*res);
Sx      = S*xc(:);
Sc      = 0.5*alpha*(Sx'*Sx);
Jc      = Dc + Sc;

% Parameters for outside visualization
para = struct('Jc',Jc,'Dc',Dc,'Sc',Sc,'omega',omega,'m',m,'Tc',xc,'Rc',Rc,'res',res(:),'nSamples',nSamples,'nCoils',nCoils);

if not(doDerivative), return; end

% Calculate gradient of T[WC]xc w.r.t. WC
for k=1:nSamples
    [~,dTx_real] = linearInter(reshape(real(xc),m),omega,y{k});
    [~,dTx_imag] = linearInter(reshape(imag(xc),m),omega,y{k}); 
    dTx{k} = dTx_real*dy{k} + 1i*dTx_imag*dy{k}; % Combine real and imaginary parts
end

% Compute the gradient and Hessian w.r.t. WC
dJ          = zeros(3, length(A)-1);
res         = reshape(res,[],nCoils,nSamples);
Hk          = cell(length(A)-1,1);
for k = 2:length(A)
    % Form the Jacobian w.r.t wc transpose times the residual where the Jacobian transpose is dTwx{k}'*S'*F'*A{k}
    dJ(:,k-1) = dTx{k}'*reshape(sum(m(1)*conj(C).*ifft2(reshape(A{k}'*res(:,:,k), [m nCoils])),3),[],1);
    
    % Compute the columns of J'*J (the approximate Hessian) for each column of 128^2 x 3 matrix dTwx{k}
    Hk{k-1} = zeros(3,3);
    Hk{k-1}(:,1) = dTx{k}'*reshape(sum(conj(C).*ifft2(reshape(A{k}'*A{k}*reshape(fft2(C.*reshape(dTx{k}(:,1),m)),[],nCoils), [m nCoils])),3),[],1);
    Hk{k-1}(:,2) = dTx{k}'*reshape(sum(conj(C).*ifft2(reshape(A{k}'*A{k}*reshape(fft2(C.*reshape(dTx{k}(:,2),m)),[],nCoils), [m nCoils])),3),[],1);
    Hk{k-1}(:,3) = dTx{k}'*reshape(sum(conj(C).*ifft2(reshape(A{k}'*A{k}*reshape(fft2(C.*reshape(dTx{k}(:,3),m)),[],nCoils), [m nCoils])),3),[],1);
end

% Put everything in correct shape for return
dJ  = real(dJ(:))';
H   = real(blkdiag(Hk{:}));

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
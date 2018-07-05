%==========================================================================
%   
%   (c) James Herring 2018/07/05
%   http://www.math.uh.edu/~herring/
%
%   Sets up and runs the coupled super-resolution problem using various 
%   objective functions
%
%   Word of warning: This whole code takes a long while to run
%
%==========================================================================

%%
%%%%%%%%%%%%%%%%%%%%%
% Setup the problem %
%%%%%%%%%%%%%%%%%%%%%

setupMoCoMRIProb;

% Set up params for all solvers

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run LAP with direct regularization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LAPd = @(y) CoupledMRIObjFctn(y, d, A, C, omega, m, 'alpha', alpha);

% Set some method parameters
maxIter     = 20;
iterSave    = true;
solver      = 'mfLAPlsdir';

tic();
[y_LAPd, his_LAPd] = GaussNewtonProj(LAPd,[x0(:); reshape(w0(:,2:end),[],1)],'solver', solver, 'solverTol',1e-2, 'maxIter', maxIter);
time_LAPd = toc();

x_LAPd = y_LAPd(1:prod(m));
w_LAPd = [zeros(3,1), reshape(y_LAPd(prod(m)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run LAP with hybrid regularization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LAPh = @(y) CoupledMRIObjFctn(y, d, A, C, omega, m, 'alpha', 0, 'regularizer', 'hybr');

% Set some method parameters
maxIter     = 20;
iterSave    = true;
solver      = 'mfLAPlsHyBR';

tic();
[y_LAPh, his_LAPh] = GaussNewtonProj(LAPh,[x0(:); reshape(w0(:,2:end),[],1)],'solver', solver, 'solverTol',1e-2, 'maxIter', maxIter);
time_LAPh = toc();

x_LAPh = y_LAPh(1:prod(m));
w_LAPh = [zeros(3,1), reshape(y_LAPh(prod(m)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run VarPro with discrete gradient regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VPd = @(w) VarproMRIObjFctn(w, d, A, C, omega, m, 'alpha',0.01,'regularizer','grad');

% Set some method parameters
maxIter     = 20;
iterSave    = true;

tic();
[w_VPd, his_VPd] = GaussNewtonProj(VPd, reshape(w0(:,2:end),[],1),'solver', [], 'maxIter', maxIter, 'iterSave', true, 'iterVP', true);
time_VPd = toc();

% Extract x with a function call
[~,para] = VPd(w_VPd);
x_VPd = para.Tc;
w_VPd = [zeros(3,1), reshape(w_VPd,3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run VarPro with discrete identity regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VPe = @(w) VarproMRIObjFctn(w, d, A, C, omega, m, 'alpha',0.01,'regularizer','eye');

% Set some method parameters
maxIter     = 20;
iterSave    = true;

tic();
[w_VPe, his_VPe] = GaussNewtonProj(VPe, reshape(w0(:,2:end),[],1), 'solver', [], 'maxIter', maxIter, 'iterSave', true, 'iterVP', true);
time_VPe = toc();

% Extract x with a function call
[~,para] = VPe(w_VPe);
x_VPe = para.Tc;
w_VPe = [zeros(3,1), reshape(w_VPe,3,[])];
%
%   This script sets up and runs the coupled super-resolution problem using
%   various objective functions
%

%%
%%%%%%%%%%%%%%%%%%%%%
% Setup the problem %
%%%%%%%%%%%%%%%%%%%%%

setup2DSuperResProb;

% Set up params for all solvers

%%
%%%%%%%%%%%%
% Run full %
%%%%%%%%%%%%
FULL = @(y) CoupledSRObjFctn(y, d, omega, mf, mc, 'alpha', alpha, 'matrixFree',0);

% Set some method parameters
maxIter = 20;
iterSave    = true;
upper_bound = [1*ones(size(x0(:))); Inf*ones(prod(size(w0(:,2:end))),1)];
lower_bound = [0*ones(size(x0(:))); -Inf*ones(prod(size(w0(:,2:end))),1)];
solver = 'mbFULLlsdir';

tic();
[y_FULL, his_FULL] = GaussNewtonProj(FULL,[x0(:); reshape(w0(:,2:end),[],1)],'upper_bound', upper_bound, 'lower_bound', lower_bound,'solver', solver, 'solverTol',1e-2, 'maxIter', maxIter);
time_FULL = toc();

x_FULL = y_FULL(1:prod(mf));
w_FULL = [zeros(3,1), reshape(y_FULL(prod(mf)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run LAP with direct regularization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LAPd = @(y) CoupledSRObjFctn(y, d, omega, mf, mc, 'alpha', alpha, 'matrixFree',1);

% Set some method parameters
maxIter = 20;
iterSave    = true;
upper_bound = [1*ones(size(x0(:))); Inf*ones(prod(size(w0(:,2:end))),1)];
lower_bound = [0*ones(size(x0(:))); -Inf*ones(prod(size(w0(:,2:end))),1)];
solver = 'mfLAPlsdir';

tic();
[y_LAPd, his_LAPd] = GaussNewtonProj(LAPd,[x0(:); reshape(w0(:,2:end),[],1)],'upper_bound', upper_bound, 'lower_bound', lower_bound,'solver', solver, 'solverTol',1e-2, 'maxIter', maxIter);
time_LAPd = toc();

x_LAPd = y_LAPd(1:prod(mf));
w_LAPd = [zeros(3,1), reshape(y_LAPd(prod(mf)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run LAP with hybrid regularization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LAPh = @(y) CoupledSRObjFctn(y, d, omega, mf, mc, 'alpha', 0, 'regularizer', 'hybr', 'matrixFree',1);

% Set some method parameters
maxIter = 20;
iterSave    = true;
upper_bound = [1*ones(size(x0(:))); Inf*ones(prod(size(w0(:,2:end))),1)];
lower_bound = [0*ones(size(x0(:))); -Inf*ones(prod(size(w0(:,2:end))),1)];
solver = 'mfLAPlsHyBR';

tic();
[y_LAPh, his_LAPh] = GaussNewtonProj(LAPh,[x0(:); reshape(w0(:,2:end),[],1)],'upper_bound', upper_bound, 'lower_bound', lower_bound,'solver', solver, 'solverTol',1e-2, 'maxIter', maxIter);
time_LAPh = toc();

x_LAPh = y_LAPh(1:prod(mf));
w_LAPh = [zeros(3,1), reshape(y_LAPh(prod(mf)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run VarPro with discrete gradient regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VPd = @(w) VarproSRObjFctn(w, d, omega, mf, mc, 'alpha', 0.01, 'matrixFree', 1);

% Set some method parameters
maxIter = 20;
iterSave    = true;
upper_bound = Inf*ones(prod(size(w0(:,2:end))),1);
lower_bound = -Inf*ones(prod(size(w0(:,2:end))),1);

tic();
[w_VPd, his_VPd] = GaussNewtonProj(VPd, reshape(w0(:,2:end),[],1),'upper_bound', upper_bound, 'lower_bound', lower_bound, 'maxIter', maxIter);
time_VPd = toc();

% Extract x with a function call
[~,para] = VPd(w_VPd);
x_VPd = para.Tc;
w_VPd = [zeros(3,1), reshape(w_VPd,3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run VarPro with discrete identity regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VPe = @(w) VarproSRObjFctn(w, d, omega, mf, mc, 'alpha', 0.01, 'regularizer', 'eye', 'matrixFree', 1);

% Set some method parameters
maxIter = 20;
iterSave    = true;
upper_bound = Inf*ones(prod(size(w0(:,2:end))),1);
lower_bound = -Inf*ones(prod(size(w0(:,2:end))),1);

tic();
[w_VPe, his_VPe] = GaussNewtonProj(VPe, reshape(w0(:,2:end),[],1),'upper_bound', upper_bound, 'lower_bound', lower_bound, 'maxIter', maxIter);
time_VPe = toc();

% Extract x with a function call
[~,para] = VPe(w_VPe);
x_VPe = para.Tc;
w_VPe = [zeros(3,1), reshape(w_VPe,3,[])];


%% 
%%%%%%%%%%%
% Results %
%%%%%%%%%%%
fprintf('norm(x0 - xtrue)/norm(x_true) = %1.4e \n', norm(x0(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w0 - wtrue)/norm(w_true) = %1.4e \n', norm(w0(:) - wtrue(:))/norm(wtrue(:)));

fprintf('norm(x_FULL - xtrue)/norm(x_true) = %1.4e \n', norm(x_FULL(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w_FULL - wtrue)/norm(w_true) = %1.4e \n', norm(w_FULL(:) - wtrue(:))/norm(wtrue(:)));

fprintf('norm(x_LAPd - xtrue)/norm(x_true) = %1.4e \n', norm(x_LAPd(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w_LAPd - wtrue)/norm(w_true) = %1.4e \n', norm(w_LAPd(:) - wtrue(:))/norm(wtrue(:)));

fprintf('norm(x_LAPh - xtrue)/norm(x_true) = %1.4e \n', norm(x_LAPh(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w_LAPh - wtrue)/norm(w_true) = %1.4e \n', norm(w_LAPh(:) - wtrue(:))/norm(wtrue(:)));

fprintf('norm(x_VPd - xtrue)/norm(x_true) = %1.4e \n', norm(x_VPd(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w_VPd - wtrue)/norm(w_true) = %1.4e \n', norm(w_VPd(:) - wtrue(:))/norm(wtrue(:)));

fprintf('norm(x_VPe - xtrue)/norm(x_true) = %1.4e \n', norm(x_VPe(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w_VPe - wtrue)/norm(w_true) = %1.4e \n', norm(w_VPe(:) - wtrue(:))/norm(wtrue(:)));
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
tolJ = 1e-4;
tolY = 1e0;
tolG = 1e-1;
maxIter     = 200;
iterSave    = true;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run LAP with direct regularization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LAPd        = @(y) CoupledMRIObjFctn(y, d, A, C, omega, m, 'alpha', alpha);

% Set some method parameters
solver      = 'mfLAPlsdir';

tic();
[y_LAPd, his_LAPd] = GaussNewtonProj(LAPd,[x0(:); reshape(w0(:,2:end),[],1)],'solver', solver, 'solverTol',1e-2, 'solverMaxIter',50,...
                                     'iterSave', iterSave, 'maxIter', maxIter, 'tolJ', tolJ, 'tolY', tolY, 'tolG', tolG);
time_LAPd   = toc();

x_LAPd      = y_LAPd(1:prod(m));
w_LAPd      = [zeros(3,1), reshape(y_LAPd(prod(m)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run LAP with hybrid regularization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LAPh = @(y) CoupledMRIObjFctn(y, d, A, C, omega, m, 'alpha', 0, 'regularizer', 'hybr');

% Set some method parameters
solver      = 'mfLAPlsHyBR';

tic();
[y_LAPh, his_LAPh] = GaussNewtonProj(LAPh,[x0(:); reshape(w0(:,2:end),[],1)],'solver', solver, 'solverTol',1e-2, 'solverMaxIter',50,...
                                     'iterSave', iterSave, 'maxIter', maxIter, 'tolJ', tolJ, 'tolY', tolY, 'tolG', tolG);
time_LAPh   = toc();
x_LAPh      = y_LAPh(1:prod(m));
w_LAPh      = [zeros(3,1), reshape(y_LAPh(prod(m)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run VarPro with discrete gradient regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VPd = @(w) VarproMRIObjFctn(w, d, A, C, omega, m, 'alpha',0.01,'regularizer','grad');

tic();
[w_VPd, his_VPd] = GaussNewtonProj(VPd, reshape(w0(:,2:end),[],1),'solver', [], 'maxIter', maxIter, 'iterSave', iterSave,...
                                   'iterVP', true, 'tolJ', tolJ, 'tolY', tolY, 'tolG', tolG);
time_VPd    = toc();

% Extract x with a function call
[~,para]    = VPd(w_VPd);
x_VPd       = para.Tc;
w_VPd       = [zeros(3,1), reshape(w_VPd,3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run VarPro with discrete identity regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VPe         = @(w) VarproMRIObjFctn(w, d, A, C, omega, m, 'alpha',0.01,'regularizer','eye');

tic();
[w_VPe, his_VPe] = GaussNewtonProj(VPe, reshape(w0(:,2:end),[],1), 'solver', [], 'maxIter', maxIter, 'iterSave', iterSave,...
                                   'iterVP', true, 'tolJ', tolJ, 'tolY', tolY, 'tolG', tolG);
time_VPe    = toc();

% Extract x with a function call
[~,para]    = VPe(w_VPe);
x_VPe       = para.Tc;
w_VPe       = [zeros(3,1), reshape(w_VPe,3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run block coordinate descent with discrete gradient regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BCDd        = @(y, flag) DecoupledMRIObjFctn(y, d, A, C, omega, m, flag, 'alpha', 0.01, 'regularizer', 'grad');

% Set some method parameters
solver      = cell(2,1); solver{1} = 'mfBCDlsdir'; solver{2} = 'mbBCDls';
solverTol   = [1e-2; 1e-2];
blocks      = [1 , length(x0(:))+1; length(x0(:)), length(x0(:))+numel(w0(:,2:end))];

tic();
[y_BCDd, his_BCDd] = CoordDescent(BCDd, [x0(:); reshape(w0(:,2:end),[],1)], blocks, 'solver', solver, 'solverTol', solverTol,...
                                  'maxIter', maxIter, 'iterSave', iterSave, 'tolJ', tolJ, 'tolY', tolY, 'tolG', tolG);
time_BCDd   = toc();
x_BCDd      = y_BCDd(1:prod(m));
w_BCDd      = [zeros(3,1), reshape(y_BCDd(prod(m)+1:end),3,[])];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run block coordinate descent with hybrid regularizer %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BCDh        = @(y, flag) DecoupledMRIObjFctn(y, d, A, C, omega, m, flag, 'regularizer', 'hybr');

% Set some method parameters
solver      = cell(2,1); solver{1} = 'mfBCDlshybr'; solver{2} = 'mbBCDls';
solverTol   = [1e-2; 1]; 
blocks      = [1 , length(x0(:))+1; length(x0(:)), length(x0(:))+numel(w0(:,2:end))];

tic();
[y_BCDh, his_BCDh] = CoordDescent(BCDh, [x0(:); reshape(w0(:,2:end),[],1)], blocks, 'solver', solver, 'solverTol', solverTol,...
                                 'maxIter', maxIter, 'iterSave', iterSave, 'tolJ', tolJ, 'tolY', tolY, 'tolG', tolG);
time_BCDh   = toc();
x_BCDh      = y_BCDh(1:prod(m));
w_BCDh      = [zeros(3,1), reshape(y_BCDh(prod(m)+1:end),3,[])];

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Relative Errors and Timings %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('norm(x0 - xtrue)/norm(x_true) = %1.4e \n', norm(x0(:) - xtrue(:))/norm(xtrue(:)));
fprintf('norm(w0 - wtrue)/norm(w_true) = %1.4e \n', norm(w0(:) - wtrue(:))/norm(wtrue(:)));

fprintf('norm(x_LAPd - xtrue)/norm(x_true) = %1.4e in %1.2secs \n', norm(x_LAPd(:) - xtrue(:))/norm(xtrue(:)), time_LAPd);
fprintf('norm(w_LAPd - wtrue)/norm(w_true) = %1.4e in %1.2secs \n', norm(w_LAPd(:) - wtrue(:))/norm(wtrue(:)), time_LAPd);

fprintf('norm(x_LAPh - xtrue)/norm(x_true) = %1.4e in %1.2secs \n', norm(x_LAPh(:) - xtrue(:))/norm(xtrue(:)), time_LAPh);
fprintf('norm(w_LAPh - wtrue)/norm(w_true) = %1.4e in %1.2secs \n', norm(w_LAPh(:) - wtrue(:))/norm(wtrue(:)), time_LAPh);

fprintf('norm(x_VPd - xtrue)/norm(x_true) = %1.4e in %1.2secs \n', norm(x_VPd(:) - xtrue(:))/norm(xtrue(:)), time_VPd);
fprintf('norm(w_VPd - wtrue)/norm(w_true) = %1.4e in %1.2secs \n', norm(w_VPd(:) - wtrue(:))/norm(wtrue(:)), time_VPd);

fprintf('norm(x_VPe - xtrue)/norm(x_true) = %1.4e in %1.2secs \n', norm(x_VPe(:) - xtrue(:))/norm(xtrue(:)), time_VPe);
fprintf('norm(w_VPe - wtrue)/norm(w_true) = %1.4e in %1.2secs \n', norm(w_VPe(:) - wtrue(:))/norm(wtrue(:)), time_VPe);

fprintf('norm(x_BCDd - xtrue)/norm(x_true) = %1.4e in %1.2secs \n', norm(x_BCDd(:) - xtrue(:))/norm(xtrue(:)), time_BCDd);
fprintf('norm(w_BCDd - wtrue)/norm(w_true) = %1.4e in %1.2secs \n', norm(w_BCDd(:) - wtrue(:))/norm(wtrue(:)), time_BCDd);

fprintf('norm(x_BCDh - xtrue)/norm(x_true) = %1.4e in %1.2secs \n', norm(x_BCDh(:) - xtrue(:))/norm(xtrue(:)), time_BCDh);
fprintf('norm(w_BCDh - wtrue)/norm(w_true) = %1.4e in %1.2secs \n', norm(w_BCDh(:) - wtrue(:))/norm(wtrue(:)), time_BCDh);

%%
%%%%%%%%%%%%%%%%%%%%%%%%
% Relative Error Plots %
%%%%%%%%%%%%%%%%%%%%%%%%

% Extract iterates for both sets of variables
x_iters_LAPd = his_LAPd.iters(1:prod(m),:);
x_iters_LAPh = his_LAPh.iters(1:prod(m),:);
x_iters_VPd  = his_VPd.iters(1:prod(m),:);
x_iters_VPe  = his_VPe.iters(1:prod(m),:);
x_iters_BCDd = his_BCDd.iters(1:prod(m),:);
x_iters_BCDh = his_BCDh.iters(1:prod(m),:);

w_iters_LAPd = [zeros(3,size(his_LAPd.array,1)); his_LAPd.iters(prod(m)+1:end,:)];
w_iters_LAPh = [zeros(3,size(his_LAPh.array,1)); his_LAPh.iters(prod(m)+1:end,:)];
w_iters_VPd  = [zeros(3,size(his_VPd.array,1)); his_VPd.iters(prod(m)+1:end,:)];
w_iters_VPe  = [zeros(3,size(his_VPe.array,1)); his_VPe.iters(prod(m)+1:end,:)];
w_iters_BCDd = [zeros(3,size(his_BCDd.iters,2)); his_BCDd.iters(prod(m)+1:end,:)];
w_iters_BCDh = [zeros(3,size(his_BCDh.iters,2)); his_BCDh.iters(prod(m)+1:end,:)];

% Make some matrices for easy evaluation
norm_xtrue     = norm(xtrue(:));
norm_wtrue     = norm(wtrue(:));

x_re_LAPd   = zeros(size(x_iters_LAPd,2),1);
x_re_LAPh   = zeros(size(x_iters_LAPh,2),1);
x_re_VPd    = zeros(size(x_iters_VPd,2),1);
x_re_VPe    = zeros(size(x_iters_VPe,2),1);
x_re_BCDd   = zeros(size(x_iters_BCDd,2),1);
x_re_BCDh   = zeros(size(x_iters_BCDh,2),1);

w_re_LAPd   = zeros(size(w_iters_LAPd,2),1);
w_re_LAPh   = zeros(size(w_iters_LAPh,2),1);
w_re_VPd    = zeros(size(w_iters_VPd,2),1);
w_re_VPe    = zeros(size(w_iters_VPe,2),1);
w_re_BCDd   = zeros(size(w_iters_BCDd,2),1);
w_re_BCDh   = zeros(size(w_iters_BCDh,2),1);

% Compute the relative errors
for k = 1:length(x_re_LAPd)
    x_re_LAPd(k)   = norm(x_iters_LAPd(:,k) - xtrue(:))/norm_xtrue;
    w_re_LAPd(k)   = norm(w_iters_LAPd(:,k) - wtrue(:))/norm_wtrue;
end

for k = 1:length(x_re_LAPh)
    x_re_LAPh(k)   = norm(x_iters_LAPh(:,k) - xtrue(:))/norm_xtrue;
    w_re_LAPh(k)   = norm(w_iters_LAPh(:,k) - wtrue(:))/norm_wtrue;
end

for k = 1:length(x_re_VPd)
    x_re_VPd(k)    = norm(x_iters_VPd(:,k) - xtrue(:))/norm_xtrue;
    w_re_VPd(k)    = norm(w_iters_VPd(:,k) - wtrue(:))/norm_wtrue;
end

for k = 1:length(x_re_VPe)
    x_re_VPe(k)    = norm(x_iters_VPe(:,k) - xtrue(:))/norm_xtrue; 
    w_re_VPe(k)    = norm(w_iters_VPe(:,k) - wtrue(:))/norm_wtrue;
end

for k = 1:length(x_re_BCDd)
    x_re_BCDd(k)   = norm(x_iters_BCDd(:,k) - xtrue(:))/norm_xtrue;
    w_re_BCDd(k)   = norm(w_iters_BCDd(:,k) - wtrue(:))/norm_wtrue;
end

for k = 1:length(x_re_BCDh)
    x_re_BCDh(k)   = norm(x_iters_BCDh(:,k) - xtrue(:))/norm_xtrue;
    w_re_BCDh(k)   = norm(w_iters_BCDh(:,k) - wtrue(:))/norm_wtrue;
end



% Plot the the relative errors
figure();
semilogy(1:length(x_re_LAPd), x_re_LAPd,'-d');
hold on; axis tight;
semilogy(1:length(x_re_LAPh), x_re_LAPh,'-h');
semilogy(1:length(x_re_VPd), x_re_VPd, '-*');
semilogy(1:length(x_re_VPe), x_re_VPe,'-+');
semilogy(1:length(x_re_BCDd), x_re_BCDd,'-s');
semilogy(1:length(x_re_BCDh), x_re_BCDh,'-^');
title('Rel. Err. Image');

figure();
semilogy(1:length(w_re_LAPd), w_re_LAPd,'-d');
hold on; axis tight;
semilogy(1:length(w_re_LAPh), w_re_LAPh,'-h');
semilogy(1:length(w_re_VPd), w_re_VPd,'-*');
semilogy(1:length(w_re_VPe), w_re_VPe,'-+');
semilogy(1:length(w_re_BCDd), w_re_BCDd,'-s');
semilogy(1:length(w_re_BCDh), w_re_BCDh,'-^');
title('Rel. Err. Motion');
legend('LAPd','LAPh', 'VPd', 'VPe', 'BCDd', 'BCDh');

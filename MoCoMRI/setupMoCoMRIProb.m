% =========================================================================
% =========================================================================
% 
%   (c) James Herring 2018/07/05
%   http://www.math.uh.edu/~herring/
%
%   Sets up 2D test data for MRI motion correction examples
%
% =========================================================================

% Configure FAIR modules
trafo('reset','trafo','rigid2D');
imgModel('reset','imgModel','linearInter');
viewImage('reset','viewImage','viewImage2Dsc','colormap','gray');

checkDataFile; 
if expfileExists , return;  end;

load xGT.mat
xtrue = flipud(double(xGT))';
xtrue = xtrue/max(abs(xtrue(:)));
C   = squeeze(double(S(:,:,:,:)));

m   = size(xtrue);
omega = [0 1 0 1];

% Generate sampling pattern
nSamples = 16;
A = getSampleA(m,nSamples,2);

% Create transformed data
wtrue = [zeros(3,1), 0.1*(rand(3,nSamples-1)-0.5)];
y       = cell(nSamples,1);
dy      = cell(nSamples,1);
Tw      = cell(nSamples,1);
grid    = getCellCenteredGrid(omega,m);
for k=1:nSamples
    [y{k},~] = trafo(wtrue(:,k),grid); % Transform points
    Tw{k} = getLinearInterMatrix(omega,m,trafo(wtrue(:,k),y{k})); % Interpolation matricesL   = getGradient(omega,m); % Did not change this at all
end
d = getAFCT(xtrue,A,C,Tw,m,'notransp');

% Add noise to data
noise_level = 0.10;
noise = randn(size(d,1),1) + 1i*randn(size(d,1),1);
noise = abs(max(d))*noise/norm(noise(:));
d = d + noise_level*noise;

% Set up initial guess for w0
w0 = wtrue + 0.01*randn(size(wtrue)); % zeros(size(wtrue)); % Takes longer to converge with zero starting guess.
grid    = getCellCenteredGrid(omega,m);
for k=1:nSamples
    [y{k},~] = trafo(w0(:,k),grid); % Transform points
    Tw{k} = getLinearInterMatrix(omega,m,trafo(w0(:,k),y{k})); % Interpolation matricesL   = getGradient(omega,m); % Did not change this at all
end

% Set up LS problem for x0
alpha = 0.01;
S = getGradient(omega,m);
rhs = [d(:); zeros(size(S,1),1)];
fop = @(x,flag) getAFCT(x(:),A,C,Tw,m,flag);
op = @(x,flag) opBlockLS(x, fop, alpha, S, flag);

[x0,~,~,~,~] = lsqr(op,rhs,1e-8,100);


% Save everything
save(expfile,'xtrue','x0','wtrue','w0','omega','m','alpha','d','A','C','nSamples');
checkDataFile;






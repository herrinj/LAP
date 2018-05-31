% =========================================================================
% (c) Lars Ruthotto 2015/12/03
% http://www.mathcs.emory.edu/~lruthot/
%
% sets up 2D test data for super resolution experiments
%
% =========================================================================

% Configure FAIR modules
distance('reset','distance','SSD');
trafo('reset','trafo','rigid2D');
imgModel('reset','imgModel','linearInter');
viewImage('reset','viewImage','viewImage2Dsc','colormap','gray');

% Check if its saved
example = 'MRISuperResolution';
checkDataFile; 
if expfileExists , return;  end;

image = @(str) double(flipud(imread(str))'); % reads and converts
plots = false;

% Set up the original data
xtrue   = image('2012-Wolters-T1.jpg');
xtrue   = xtrue./max(xtrue(:));
mf      = [128 128]; % dimension for fine resolution data
mc      = [32 32];   % dimension for coarse resolution data
omega   = [-100 100 -140 140];
xtrue   = reshape(linearInter(xtrue,[-80 80 -120 120],getCellCenteredGrid(omega,mf)),mf);
omega   = [-16 16 -16 16]; % image domain
hc      = (omega(2:2:end)-omega(1:2:end))./mc;
hf      = (omega(2:2:end)-omega(1:2:end))./mf;
nVol    = 32;              % number of coarse resolution frames


wtrue = []; d  = []; wt = [0;0;0];
grid        = getCellCenteredGrid(omega,mc); % cell-centered grid for transformations
noise_level = 0.02; % noise level
rng(4);
for i=1:nVol 
    % Generate noisy data
    yc = trafo(wt,grid);
    dt = linearInter(xtrue,omega,yc);
    noise = randn(size(dt));
    noise = noise_level*norm(reshape(dt,[],1))*noise/norm(noise(:));
    d = [d, dt + noise]; 
    % Generate true motion parameters
    wtrue = [wtrue, wt];
    wt =  [.2;2;2].*randn(3,1);
end
d = reshape(d,[mc nVol]);


% Rigid registration to first frame for w0
w0 = zeros(3,nVol);
for i=2:nVol
   fprintf('\nRegistering volume %d of %d to first volume\n\n',i,nVol);
   MLdata = getMultilevel({d(:,:,1),d(:,:,i)},omega,mc,'fig',0);  
   w0(:,i)  = MLPIR(MLdata,'plots',plots,'minLevel',4);
end

% Solve linear problem for x0

% Interpolation and transformation matrix
A = [];
for i=1:nVol
    yi = trafo(w0(:,i),grid);
    Ai = getLinearInterMatrix(omega,mf,yi);
    A  = [A; Ai];
end

% Set up LS problem
alpha   = 1e-2; 
S       = sqrt(prod(hf))*getGradient(omega,mf); 
op      = [sqrt(prod(hc))*A; sqrt(alpha)*S]; 
rhs     = [sqrt(prod(hc))*d(:); zeros(size(S,1),1)];

% Solver the LS problem
myrank  = 20;
[U,B,V] = lancBiDiag(op,rhs,myrank); % No reorthogonalization 
[Ub,Sb,Vb]   = svd(full(B));
U = U*Ub; U = U(:,1:myrank); 
Sb = Sb(1:myrank,1:myrank);
V = V*Vb;
x0 = V*(Sb\(U'*rhs));
x0(x0<0) = 0;
x0(x0>1) = 1;

% Save everything
save(expfile,'xtrue','x0','wtrue','w0','omega','mf','mc','alpha','d','nVol');
checkDataFile;
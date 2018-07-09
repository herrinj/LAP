function [yc,his] = CoordDescent(fctn,y0,blocks,varargin)
%
% function [yc,his] = GaussNewtonProj(fctn,yc,varargin)
%
% Block coordinate descent with projected Gauss-Newton step for each block
% of variables with Projected Armijo line search. Minimizes J = fctn(yc,j)
% 
% Input:
%   fctn      - function handle, should take calls fctn(yc,j) where yc is
%               current solution and j is flag to return the 
%               gradient/Jacobian/Hessian w.r.t jth block
%   y0        - starting guess (required), should be within feasible region
%   blocks    - 2xN array with N = blocks of variables and block(1,j) the 
%               and block(2,j) the index of the start and end of the jth 
%               block
%   varargin  - optional parameter, see below
%
% Output:
%   yc        - numerical optimizer (current iterate)
%   his       - iteration history
%
%==============================================================================

% Block coordinate descent parameters
numBlocks    = size(blocks,2);
maxIter      = 50;
yStop        = [];
Jstop        = [];
paraStop     = [];
tolJ         = 1e-4;            % stopping tolerance, objective function
tolY         = 1e-4;            % stopping tolerance, norm of solution
tolG         = 1e-4;            % stopping tolerance, norm of gradient

% Gauss-Newton step solve parameters, can vary for each block
solver       = cell(numBlocks,1);             
solverMaxIter= 50*ones(numBlocks,1);              
solverTol    = 1e-1*ones(numBlocks,1);

% Bound constraints
upper_bound  = Inf*ones(numel(y0),1); % only accepts bounds on real variables 
lower_bound  = -Inf*ones(numel(y0),1);

% Line search parameters
lsMaxIter    = 10;           % maximum number of line search iterations
lsReduction  = 1e-4;         % reduction constant for Armijo condition
lineSearch   = @proj_armijo; % Could potentially call to other projected LS

% Method options
verbose      = true;         % flag to print out
iterSave     = 0;            % flag to save iterations
stop         = zeros(5,1);   % vector for stopping criteria
Plots        = @(iter,para) []; % for plots;

% History output
his          = [];
hisArray     = zeros(maxIter*numBlocks+1,8);
hisStr       = {'iter','block','J','Jold-J','|proj_dJ|','|dy|','LS','Active'};


% Overwrite default parameters above using varargin
for k=1:2:length(varargin)     
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

% Evaluate objective function for stopping criteria
if isempty(yStop) 
    yStop = y0; 
end

if (isempty(Jstop) || isempty(paraStop))
    [Jstop,paraStop] = fctn(yStop, 0); Jstop = abs(Jstop) + (Jstop == 0); 
    Plots('stop',paraStop);
end

yc = y0;
active = (yc <= lower_bound)|(yc >= upper_bound);
[Jc,para,dJ,~] = fctn(yc,0); 
proj_dJ = proj_grad(dJ, yc, lower_bound, upper_bound);
Plots('start',para);
iter = 0; yOld = 0*yc; Jold = Jc;
hisArray(1,:) = [0 , 0, Jc, Jc, norm(proj_dJ), 0, 0, sum(active>0)];


% Save iterates
if iterSave
    iterArray      =  zeros(numel(yc), maxIter+1);
    iterArray(:,1) = yc;
end

% Print stuff
if verbose
    fprintf('%s %s %s\n',ones(1,20)*char('='),mfilename,ones(1,20)*char('='));
    fprintf('[ maxIter=%s / tolJ=%s / tolU=%s / tolG=%s / length(yc)=%d ]\n',...
    num2str(maxIter),num2str(tolJ),num2str(tolY),num2str(tolG),length(yc));
    fprintf('%4s %4s %-12s %-12s %-12s %-12s %4s %-8s \n %s \n', hisStr{:},char(ones(1,64)*'-'));
    dispHis = @(var) fprintf('%4d %4d %-12.4e %-12.3e %-12.3e %-12.3e %4d %-8d \n',var);
    dispHis(hisArray(1,:));
end

% Start projected Gauss-Newton iteration
while 1
   
    % Check stopping criteria
    stop(1) = (iter>0) && abs(Jold-Jc)   <= tolJ*(1+abs(Jstop));
    stop(2) = (iter>0) && (norm(yc-yOld) <= tolY*(1+norm(y0)));
    stop(3) = norm(proj_dJ)              <= tolG*(1+abs(Jstop));
    stop(4) = norm(proj_dJ)              <= 1e6*eps;
    stop(5) = (iter >= maxIter);
    if (all(stop(1:3)) || any(stop(4:5)))
        break;  
    end
    
    iter = iter+1;  
    
    for k = 1:numBlocks % Inner loop performs single projected Gauss-Newton step over each block of variables
        
        % Indices and arrays needed to update the kth block
        blockStart = blocks(1,k);
        blockEnd   = blocks(2,k);
        blockYc    = yc(blockStart:blockEnd); % yc on current block
        blockActive= active(blockStart:blockEnd); % active set on block
        blockLB    = lower_bound(blockStart:blockEnd); % lower_bound on block
        blockUB    = upper_bound(blockStart:blockEnd); % upper_bound on block

        % Evalute objective function on current block 
        [Jc,para,blockDJ,blockOp] = fctn(yc,k);     
       
        % Gauss-Newton step on inactive set
        [dy_in, solverInfo] = stepGN(blockOp, -blockDJ, solver{k}, 'active', blockActive, 'solverMaxIter',solverMaxIter(k), 'solverTol', solverTol(k));

        % Pull out updated gradient (only for hybrid regularizer)
        if isfield(solverInfo,'dJ')
            blockDJ = solverInfo.dJ;
        end
        dJ = zeros(size(yc));
        dJ(blockStart:blockEnd) = blockDJ;
        proj_dJ = proj_grad(dJ, yc, lower_bound, upper_bound);

        % Projected gradient descent on active set
        dy_act = zeros(size(blockYc));
        dy_act(blockYc == blockLB) = -dJ(blockYc == blockLB); 
        dy_act(blockYc == blockUB) = -dJ(blockYc == blockUB);    

        % Combine the steps
        if sum(blockActive)==0
            nu = 0;
        else
            nu = max(abs(dy_in))/max(abs(dy_act)); % scaling factor (Haber, Geophysical Electromagnetics, p.110)
        end
        dy = zeros(size(yc));
        dy(blockStart:blockEnd) = dy_in + nu*dy_act;

        % Line search
        [yt, exitFlag, lsIter] = lineSearch(fctn, yc, dy, Jc, proj_dJ, lower_bound, upper_bound, lsMaxIter, lsReduction); 
        if exitFlag==0
            break;
        end

        % Save new values and re-evaluate objective function
        yOld = yc; Jold = Jc;
        yc = yt; % update iterate
        active = (yc <= lower_bound)|(yc >= upper_bound);      % update active set
        [Jc,para,dJ] = fctn(yc,0);                             % evalute full objective function for metrics
        proj_dJ = proj_grad(dJ, yc, lower_bound, upper_bound); % projected gradient

        % Some output
        hisArray((iter-1)*numBlocks+k+1,:) = [iter, k, Jc, Jold-Jc, norm(proj_dJ), norm(yc-yOld), lsIter, sum(active>0)];
        if verbose
            dispHis(hisArray((iter-1)*numBlocks+k+1,:));
        end
        
        para.normdY = norm(yc - yOld);
        Plots(iter,para);
    end 
    
    if iterSave % only save iteration after full iteration
        iterArray(:,iter+1) = yc;
    end
end

Plots(iter,para);

% Clean up and output
his.str = hisStr;
his.array = hisArray(1:(iter-1)*numBlocks+k+1,:);
if iterSave
   his.iters = iterArray(:,1:iter+1); 
end

if verbose
    fprintf('STOPPING:\n');
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(1),...
    '(Jold-Jc)',(Jold-Jc),'tolJ*(1+|Jstop|)',tolJ*(1+abs(Jstop)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(2),...
    '|yc-yOld|',norm(yc-yOld),'tolY*(1+norm(yc)) ',tolY*(1+norm(yc)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(3),...
    '|dJ|',norm(proj_dJ),'tolG*(1+abs(Jstop))',tolG*(1+abs(Jstop)));
    fprintf('%d[ %-10s=%16.8e <= %-25s=%16.8e]\n',stop(4),...
    'norm(dJ)',norm(proj_dJ),'eps',1e3*eps);
    fprintf('%d[ %-10s=  %-14d >= %-25s=  %-14d]\n',stop(5),...
    'iter',iter,'maxIter',maxIter);

    FAIRmessage([mfilename,' : done !'],'=');
end
    
end

function [yt, exitFlag, iter] = proj_armijo(fctn, yc, dy, Jc, proj_dJ, lower_bound, upper_bound, lsMaxIter, lsReduction) 
%
%   This function peforms a projected Armijo line search obeying the bounds
%   
%   Input:      ftcn - objective function from proj. Gauss-Newton iteration
%                 yc - current iterate
%                 dy - update direction for currant iterate
%                 Jc - current objective function value
%            proj_dJ - projected gradient
%       
%          lsMaxIter - maximum number of line search iterations
%        lsReduction - required reduction for Armijo condition
%   
%   Output:       yt - updated iterate yc + t*dy
%           exitFlag - flag, 0 failure, 1 success
%               iter - number of line search iterations
%

t = 1; % initial step 1 for Gauss-Newton
iter = 1;
cond = zeros(2,1);

while 1
    yt = yc + t*dy;
    active = (yc <= lower_bound)|(yc >= upper_bound);
    if sum(active)>0
        yt = min(max(yt,lower_bound),upper_bound); % Only impose bounds on the image
    end
    Jt = fctn(yt,0);
    
    % check Armijo condition
    cond(1) = (Jt<Jc + t*lsReduction*(reshape(proj_dJ,1,[])*dy));
    cond(2) = (iter >=lsMaxIter);
    
    if cond(1)
        exitFlag = 1;
        break; 
    elseif cond(2)
        exitFlag = 0;
        fprintf('Line search fail: maximum iterations = %d \n',lsMaxIter);
        break;
    end
        
    t = t/2; % sterp reduction factor
    iter = iter+1;
end
end

function [proj_dJ] = proj_grad(dJ, yc, lower_bound, upper_bound)
% 
%   This function projects the gradient
%
%   Input:      dJ - unprojected gradient
%               yc - current image and motion parameters to detect which 
%                    variables are on the bounds
%      lower_bound - vector containing elementwise lower bounds on yc
%      upper_bound - vector containint elementwise upper bounds on yc
%
%   Output:
%          proj_dJ - projected gradient
%

proj_dJ = dJ(:);
proj_dJ(yc == lower_bound) = min(dJ(yc == lower_bound),0);
proj_dJ(yc == upper_bound) = max(dJ(yc == upper_bound),0);

end

function [f,para,df,d2f] = Rosenbrock(x)
x = reshape(x,2,[]);
para = [];
f = (1-x(1,:)).^2 + 100*(x(2,:) - (x(1,:)).^2).^2;

if nargout>1 && size(x,2)==1
    df = [2*(x(1)-1) - 400*x(1)*(x(2)-(x(1))^2); ...
        200*(x(2) - (x(1))^2)];
end

if nargout>2 && size(x,2)==1
    n= 2;
    d2f=zeros(n);
    d2f(1,1)=400*(3*x(1)^2-x(2))+2; d2f(1,2)=-400*x(1);
    for j=2:n-1
        d2f(j,j-1)=-400*x(j-1);
        d2f(j,j)=200+400*(3*x(j)^2-x(j+1))+2;
        d2f(j,j+1)=-400*x(j);
    end
    d2f(n,n-1)=-400*x(n-1); d2f(n,n)=200;
end
end

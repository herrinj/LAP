function [A] = getSampleA(m, nsweep, flag)
%
%   [A] = getSampleA(m, nsweeps, flag)
%
%   This function takes a set of dimensions n, a number of sweeps, and a flag
%   indicating a sampling pattern and creates a cell array A with nsweeps
%   cells. The kth cell of A corresponds to the sampling pattern of the kth
%   data frame in  MRI motion correction problem.
%
%   Input:        m - dimension of problem
%           nsweep - number of sweeps for sampling, should be divisible by 4
%              flag - sampling pattern
%                       0,'CS'   - Cartesian sequential 
%                       1,'CP1'  - Cartesian parallel 1D
%                       2,'CP2'  - Cartesian parallel 2D <--- Best
%                       3,'RND'  - Random                <--- Best
%                       4,'RS'   - Radial sequential 
%
%   Output:       A - cell array with nsweeps cells 
%       


Id  = speye(prod(m));
A = cell(nsweep,1);

switch flag
    case{'CS',0}
        cnt = 0;
        for k = 1:length(A)
            A{k} = sparse(Id(cnt +(1:prod(m)/nsweep),:));
            cnt = cnt + prod(m)/nsweep;
        end
    case{'CP1',1}
        for k = 1:length(A)
            A{k} = sparse(Id(:,k:nsweep:end))';
        end
    case{'CP2',2} 
        for k = 1:length(A)
            vec = zeros(1,m(2));
            vec(k:nsweep:end) = 1;
            mat = toeplitz([vec(1) fliplr(vec(2:end))], vec);
            inds = find(mat > 0);
            A{k} = sparse(Id(inds,:));
        end
    case{'RND',3}
        [~,inds] = sort(rand(prod(m),1));
        cnt = 0;
        for k = 1:length(A)
            A{k} = sparse(Id(sort(inds(cnt + (1:prod(m)/nsweep))),:));
            cnt = cnt + prod(m)/nsweep;
        end
    case{'RS',4}
        n = sqrt(nsweep);
        cnt = 0;
        for j = 1:n
            for i = 1:n
                temp = zeros(m);
                temp(1 + (i-1)*m(1)/n: i*m(1)/n, 1 + (j-1)*m(2)/n : j*m(2)/n) = 1.0;
                inds = find(temp > 0);
                A{i + n*(j-1)} = sparse(Id(inds,:));
            end
        end
end

end


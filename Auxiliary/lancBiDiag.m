function [U,B,V] = lancBiDiag(A,d,k,m,n)
%
%   Performs Lanczos (Golub-Kahan) bidiagonalization of rank k for an 
%   operator A given a starting vector d. The bidiagonalization satisfies 
%   the relationships:
%
%   U'*A*V = B and A*V = U*B and A'*U = V*B' 
%
%   Input:  A - operator as matrix or function handle
%           d - vector to initialize subspace
%           k - rank of bidiagonalization to perform
%           m - rows in operator
%           n - columns in operator
%           
%   Output: U - m x (k+1) orthogonal matrix
%           B - (k+1) x k bidiagonal matrix
%           V - n x k orthogonal matrix
%
%   Ref: p.571 of Golub and Van Loan, Matrix Computations, SIAM 2013   
%
if isa(A,'function_handle')
    
    U = zeros(m,k); V = zeros(n,k);
    
    % Prepare for Lanczos iteration.
    v = zeros(n,1);
    beta = norm(d); u = d/beta;
    U(:,1) = u;
    
    for i=1:k
        r = A(u,'transp') - beta*v;
        for j=1:i-1, r = r - (V(:,j)'*r)*V(:,j); end
        alpha = norm(r);
        v = r/alpha;
        B(i,2) = alpha;
        V(:,i) = v;
        p = A(v,'notransp') - alpha*u;
        %for j=1:i-1, p = p - (U(:,j)'*p)*U(:,j); end % Reothogonalization
        beta = norm(p); u = p/beta;
        B(i,1) = beta;
        U(:,i+1) = u;
    
    end
    B = spdiags(B,[-1,0],k+1,k);
else
    [m,n] = size(A);  
    U = zeros(m,k); V = zeros(n,k);
    % Prepare for Lanczos iteration.
    v = zeros(n,1);
    beta = norm(d); u = d/beta;
    U(:,1) = u;
    for i=1:k
        r = (A'*u) - beta*v;
        for j=1:i-1, r = r - (V(:,j)'*r)*V(:,j); end
        alpha = norm(r);
        v = r/alpha;
        B(i,2) = alpha;
        V(:,i) = v;
        p = A*v - alpha*u;
        %for j=1:i-1, p = p - (U(:,j)'*p)*U(:,j); end % Reothogonalization
        beta = norm(p); u = p/beta;
        B(i,1) = beta;
        U(:,i+1) = u;
    end
    B = spdiags(B,[-1,0],k+1,k);
end
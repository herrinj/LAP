function out_vec = opBlockLS(in_vec, A, alpha, S, flag)
%
%   Returns matrix-vector products
%       out_vec = [A; sqrt(alpha)*S]*in_vec 
%   and transpose for matrix-free operator A and matrix based regularizer S
%
switch flag
    case{'notransp'}
        out_vec = [A(in_vec,'notransp'); sqrt(alpha)*(S*in_vec)];
    case{'transp'}
        dim = size(S,1);
        x1 = in_vec(1:length(in_vec)-dim);
        x2 = in_vec(length(in_vec)-dim+1:end);
        out_vec = A(x1,'transp') + sqrt(alpha)*(S'*x2);
end

end

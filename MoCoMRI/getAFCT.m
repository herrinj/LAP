function out_vec = getAFCT(in_vec, A, C, T, m, flag)
%
%   Performs forward and transpose operator for MRI Motion Correction
%   Forward operator given by:
%
%   out_vec = A*F*S*T(w)*in_vec
%   
%   where:  A - cell array containing sampling pattern for k samples
%           F - 2D FFTs
%           C - 3D array of coil sensitivities 
%           T - cell array containing transformation for k samples
%

sampleSize  = size(A{1},1); % Assumes all samples same size!
nSamples    = length(A);
nCoils      = size(C,3);
switch flag
    case {'notransp'}
        out_vec = zeros(sampleSize*nCoils,nSamples);
        
        for k= 1:nSamples
            t1 = reshape(T{k}*in_vec(:),m);
            t2 = C.*t1;
            t3 = reshape(fft2(t2),[],nCoils)/sqrt(prod(m));
            out_vec(:,k) = reshape(A{k}*t3,[],1);
        end
        
        out_vec = out_vec(:);
        
    case {'transp'}
      
        out_vec = zeros(prod(m),1);
        in_vec = reshape(in_vec, [], nCoils, nSamples); 
        
        for k = 1:nSamples
            t1 = A{k}'*in_vec(:,:,k);
            t2 = sqrt(prod(m))*ifft2(reshape(t1, [m nCoils]));
            t3 = sum(conj(C).*t2,3);
            t4 = T{k}'*t3(:);
            out_vec = out_vec + t4;
        end
end
end


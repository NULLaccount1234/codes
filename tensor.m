%tensor_ class for 3d tensor_s
% MIT License
% 
% Copyright (c) 2025, Bowen
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

%tensor_ methods:
%   1.set method
%   2.plus  
%   3.minus
%   4.t-product (2d case is matrix multiplication)
%   5.reshahpe
%   6.scalar product (operator overloading ".*")
%   7.fast t-product based on fft (operator overloading "*"
%   8.subref
%   9.transpose in tensor_ setting. (operator overloading ')
%   10.t-svd
%   11.t-trancated_svd
%   12.subsasgn (operator overloading A(I) = B(I))
%   13.Moore-Penrose inverse
%   14.dot product
% . 15.tubal rank (rankt)
%%   subsref and subsasgn method only support rectangular index or linear indexing
%%   currently. And such methods will be improved more later.
%
%%   More will be updated soon:
%%   12.t-cur approximation by random sampling.
%%   13.nuclear norm
%   
%   
%%%%%%% Usage example:
%       >> A = tensor_(rand(3,4,3));
%       >> isa(A,'tensor_');
%       >> B = tensor_(rand(4,5,3));
%       >> C = tproduct(A,B);
%       >> C = A * B (fast t-product based on FFT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef tensor_
    properties
        data

    end
    properties (Dependent = true)
        shape
    end

    methods
        %constructor
        function this = tensor_(data)
            if nargin == 0
                warning('no data!You could start with command like tensor_(rand(2,3,4));')
            else
    
            this.data = data;
            end
        end
        function shape = get.shape(this)
            shape = size(this.data);
        end
        

        function this = set(this,new_data)
           %%% set
           %%%%%%%%%%%%%usage example%%%%%%%%%%%%%%%%
           %  A = tensor_(rand(2,2));                %
           %  set(A, rand(3,3)); <-- reset tensor_ A.%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           this.data = new_data;
          
        end
        function value = norm(this) 
            %%% 2-norm (will be updated soon )
            value = norm(this.data,'fro');
        end
        function this = plus(this,another)
            %%% +
            %%%%%%%%%%%%usage explanation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  operator overloading in matlab                         %
            %  A = tensor_(rand(2,2));                                 %
            %  B = tensor_(rand(2,2));                                 %
            %  A + B will not change neither A or B;                  %
            %  if users want to get result A + B, users could do like %
            %  C = A +B or C = A.plus(B);                             %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if ~isa(another,'tensor_')
                error('add number is not a tensor_');
            end
            if ~isequal(this.shape,another.shape)
                error('plus should be the same size');
            end
            this.data = this.data + another.data;
        end
        function this = minus(this,another)
            %%% -
            %%%%%%%%%%%%usage explanation%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  operator overloading in matlab                       %
            %  A = tensor_(rand(2,2));                               %
            %  B = tensor_(rand(2,2));                               %
            %  A - B will not change neither A or B;                %
            %  if user want to get result A - B, user could do like %
            %  C = A - B or C = A.minus(B);                         %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if ~isa(another,'tensor_')
                error('add number is not a tensor_');
            end
            if ~isequal(this.shape,another.shape)
                error('plus should be the same size');
            end
            this.data = this.data - another.data;
        end
        function this = uminus(this)
            %%%%%%%%%%%%usage explanation%%%%%%%%%%%%%%%
            %  operator overloading in matlab          %
            %  A = tensor_(eye(2));                     %
            %  - A : [-1 0; 0 -1];                     %
            %  Notice it will not change A itself      %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            this.data = -this.data;
        end
        
        function this = times(varargin)
            %%% scalar product
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % operator overloading in matlab           %
            % >> A = tensor_(eye(2));                   %
            % >> 2*A                                   %
            % Notice this API will not change A.data   %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if (nargin ~= 2)
                error('scalar production is binary operation')
            end
            if isscalar(varargin{1}) && isa(varargin{2},'tensor_')
                this = tensor_(varargin{1}.* varargin{2}.data);
                
            elseif isscalar(varargin{2}) && isa(varargin{1},'tensor_')
                this = tensor_(varargin{2}.* varargin{1}.data);
            else
                error('please check wheter one is scalar and another is a tensor_.')
            end

        end
        
        function this = tproduct(this, another)
           %%% t-tensor_ product(slow)
           %1d tensor_(scalar)
           if isequal(ndims(this.data),1) && isequal(ndims(another.data),1)
                this.data = this.data * another.data;
           %2d tensor_(matrix)
           elseif isequal(ndims(this.data),2) && isequal(ndims(another.data),2)
                this.data = this.data * another.data;
           %3d tensor_. Implement by "Circlant matrix * MatVec"
           elseif isequal(ndims(this.data),3) && isequal(ndims(another.data),3)
                [n1,n2,n3] = size(this.data); 
                [n2_,n4,n3_] = size(another.data);
                warning('if the number of third dimension is 1, Matlab will degenerate it to matrix!')
                if ~isequal(n2,n2_)|| ~isequal(n3,n3_)
                    error('can not do t-product, please check two tensor_ size/shape.')
                end
                % circ(this.data)
                % the code for circulant was reference by the following link
                % https://www.mathworks.com/matlabcentral/answers/521456-how-to-wirte-an-efficient-program-to-compute-the-block-circulant-matrix-of-a-3-d-array
                circ = zeros(n1*n3, n2*n3);
                X = permute(this.data, [1 3 2]);
                X = reshape(X, n1*n3, n2);
                for i=1:n3
                    circ(:,(i-1)*n2+1:i*n2) = circshift(X, (i-1)*n1);
                end
                %MatVec(another.data)
                Matvec = zeros(n2*n3,n4);
                for i = 1:n3
                    Matvec((i-1)*n2+1:i*n2,:) = another.data(:,:,i);
                end
                result = circ * Matvec;
                this.data = zeros(n1,n4,n3);
                for i = 1:n3
                    this.data(:,:,i) = result((i-1)*n1+1:i*n1,:);
                end
           else
               error(['higher dimensional cases will be updated soon,\' ...
                   'please check whether illegal to product if ndims <=3'])
           end
    
        end
        function this = mtimes(this, another)
           %%% fast t-tensor_ product
           %1d tensor_(scalar)
           if isequal(ndims(this.data),1) && isequal(ndims(another.data),1)
                this.data = this.data * another.data;
           %2d tensor_(matrix)
           elseif isequal(ndims(this.data),2) && isequal(ndims(another.data),2)
                this.data = this.data * another.data;
           %3d tensor_. Implement by "Circlant matrix * MatVec"
           elseif isequal(ndims(this.data),3) && isequal(ndims(another.data),3)
                [n1,n2,n3] = size(this.data); 
                [n2_,n4,n3_] = size(another.data);
                %warning('if the number of third dimension is 1, Matlab will degenerate it to matrix!')
                if ~isequal(n2,n2_)|| ~isequal(n3,n3_)
                    error('can not do t-product, please check two tensor_ size/shape.')
                end
                tmp_x = fft(this.data,[],3);
                tmp_y = fft(another.data,[],3);
                this.data = zeros(n1,n4,n3);
                for i = 1:n3
                    this.data(:,:,i) = tmp_x(:,:,i)*tmp_y(:,:,i);
                end
                this.data = ifft(this.data,[],3);
           else
               error(['higher dimensional cases will be updated soon,\' ...
                   'please check whether illegal to product if ndims <=3'])
           end
    
        end
        function bool = eq(this,another)
            %%% Equality ==
            if ~isa(another, 'tensor_')
                error('both data need to be tensor_ type.');
            end
            if isequal(this.data,another.data)
                bool = 1;
            else
                bool = 0;
            end
        end
        function this = reshape(this,siz)
            %%% reshape
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % usage example:                                             %
            %     >> A = tensor_(rand(2,2));                              %
            %     >> B = reshape(A,[1,4]);                               %
            %     >> B = A.reshape([1,4]);                               %
            %     such API does not change layout(shape) of A;           %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if prod(this.shape) ~= prod(siz)
               
                error('can not reshape due to inconsisitence of shape')
            end
            this.data = reshape(this.data,siz);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function this = subsref(this,s)
            %%%3d tensor_ index. 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%  
            %%% >>> A = tensor_(rand(2,2));                                %
            %%% >>> B = A(:,:,[1,2]);                                     %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % warning(['currently we only support index for 3d tensor_\' ...
        %                         'index information along all 3 dimension are required'\ ...
        %                         'only rectangular form and linear indexing are' ...
        %                         ' supported.']);
            switch s(1).type
                case '{}'
                    error('Cell contents reference is not available currently.')
                case '.'
                    fieldname = s(1).subs;
                    switch fieldname
                        case 'data'
                            this = this.data;
                        case 'shape'
                            this = this.shape;
                        case 'norm'
                            this = norm(this.data);
                    end       

                case '()'

                    if (numel(s(1).subs) == numel(this.shape))
                        region = s(1).subs;

                        % Extract the data
                        this = tensor_(this.data(region{:}));
                    else 
                        
                        idx = s(1).subs{1};
                        this = tensor_(this.data(idx));

                    end   
            end
        end
        function res = ctranspose(this)
        %%% only support 3d tensor_ transpose temporarily
        %%% operator overloading to complex conjugate transpose
            if ndims(this.data)~= 3
                error('Only support 3d tensor_ transpose temporarily.')
            end
            [n1,n2,n3] = size(this.data);
            Z = this.data;
            res = zeros(n2,n1,n3);
            res(:,:,1) = Z(:,:,1)';
            for i = 2:n3
                res(:,:,i) = Z(:,:,n3+2-i)';
            end
            res = tensor_(res);
        end
        function [U,S,V] = tsvd(this)
        %%%only support 3d tensor_ svd temporarily.
        %%%U*S*transpose(V) = this.data
            if ndims(this.data)~= 3
                error('Only support 3d tensor_ svd temporarily.')
            end
            [n1,n2,n3] = size(this.data);
            Z = fft(this.data,[],3);
            U = zeros(n1,n1,n3);
            S = zeros(n1,n2,n3);
            V = zeros(n2,n2,n3);
            for i = 1:n3
                [U_tmp,S_tmp,V_tmp] = svd(Z(:,:,i));
                U(:,:,i) = U_tmp;
                S(:,:,i) = S_tmp;
                V(:,:,i) = V_tmp;
            end
            U = tensor_(ifft(U,[],3));
            S = tensor_(ifft(S,[],3));
            V = tensor_(ifft(V,[],3));
    
        end
        function [U,S,V] = trsvd(this,k)
        %%%only support 3d tensor_ trancated svd temporarily.
        %%%U*S*transpose(V) = this.data
            if ndims(this.data)~= 3
                error('Only support 3d tensor_ svd temporarily.')
            end
            [n1,n2,n3] = size(this.data);
            Z = fft(this.data,[],3);
            U = zeros(n1,k,n3);
            S = zeros(k,k,n3);
            V = zeros(n2,k,n3);
            for i = 1:n3
                [U_tmp,S_tmp,V_tmp] = svds(Z(:,:,i),k);
                %size(U_tmp)
                U(:,:,i) = U_tmp;
                S(:,:,i) = S_tmp;
                V(:,:,i) = V_tmp;
            end
            U = tensor_(ifft(U,[],3));
            S = tensor_(ifft(S,[],3));
            V = tensor_(ifft(V,[],3));
    
        end
        function this = subsasgn(this,s,b)
            %%%operator overloading of subsasgn
            if s.type == '()'
                if numel(s.subs)==3

                    this.data(s.subs{:},1) = b.data;
                    %this = tensor_(this);
                else
                    idx = s.subs{1};
                    if any(idx > numel(this.data))
                        error('Index overflow!')
                    end
                    idx = unique(idx);
                    this.data(idx) = b.data;
                    
                end
            end
        end
        function res = mpinv(this)
            if ndims(this.data)~= 3
                error('currently we only support moore-penrose inverse for 3d tensor_')
            end
            [x_siz,y_siz,z_siz] = size(this.data);
            res =zeros(y_siz,x_siz,z_siz);
            tmp = fft(this.data,[],3);
            for i = 1:z_siz
                res(:,:,i) = pinv(tmp(:,:,i));
            end
            res = ifft(res,[],3);
            res = tensor_(res);
        end
        function value = dot(this,another)
            if ~isequal(size(this.data),size(another.data))
                error('shape are not same for dot production')
            end
            value = sum(dot(this.data,another.data),'all');

        end
        function r = rankt(this)
            [n1,n2,n3] = size(this.data);
            hat = fft(this.data,[],3);
            [u,s,v] = pagesvd(hat);
            r = 0;
            for i = 1:n3
                if r<rank(s(:,:,i))
                    r = rank(s(:,:,i));
                end
            end
        end

end
end

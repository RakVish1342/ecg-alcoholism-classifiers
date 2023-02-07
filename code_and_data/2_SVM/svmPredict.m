function pred = svmPredict(model, X)

% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example
if (size(X, 2) == 1)
    % Examples should be in rows
    X = X';
end

% Dataset 
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

if strcmp(func2str(model.kernelFunction), 'linearKernel')
    % We can use the weights and bias directly if working with the 
    % linear kernel
    p = X * model.w + model.b;
elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X1 = sum(X.^2, 2); % kxn ---> kx1 ... here X is nothing but Xval 
    X2 = sum(model.X.^2, 2)'; % size(idx ~= 0) x m .... say txn (ie considering 't' SVs) ---> tx1
    K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X')); % kxn::(txn)'
    K = model.kernelFunction(1, 0) .^ K; % kxt ... Kernel for all pairs taken from the k Xvalidation set and the 't' SVs (support vectors)  
    
    
    
    % I had edditted it to this section to make it seem more meaningful
    % p = sum (alphas * model.y * Kernel)
    tmp1 = bsxfun(@times, model.y', K); %1xn
    tmp2 = bsxfun(@times, model.alphas', tmp1); %1xn
    p = sum(tmp2, 2); % a single number
    
    
    
else
    % Other Non-linear kernel
    for i = 1:m
        prediction = 0;
        for j = 1:size(model.X, 1)
            prediction = prediction + ...
                model.alphas(j) * model.y(j) * ...
                model.kernelFunction(X(i,:)', model.X(j,:)');
        end
        p(i) = prediction + model.b;
    end
end

% Convert predictions into 0 / 1
pred(p >= 0) =  1;
pred(p <  0) =  0;

end


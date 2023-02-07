function sim = gaussianKernel(x1, x2, sigma)

    % Ensure that x1 and x2 are column vectors
    x1 = x1(:); x2 = x2(:);

    % You need to return the following variables correctly.
    sim = 0;

    % x1 eqt to xi and x2 eqt to xj

    nr = sum( (x1 - x2).^2);

    sim = exp (-nr/(2*(sigma*sigma)));
    
end

function X_norm = normalize_no_save(X)
  
  X_norm = zeros(size(X));
  for i = 1:size(X, 2)
    feature = X(:,i);
    X_norm(:,i) = ( feature-min(feature) )/( max(feature) - min(feature) );
  end
%  csvwrite(filename, X_norm);
end
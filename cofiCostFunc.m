function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% return the cost and gradient

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


%Cost Function
J = 1/2 * sum(sum((X * Theta' .* R - Y .* R).^2)) + lambda/2 *sum(sum(Theta .^ 2)) + ...
	lambda/2 * sum(sum(X .^ 2));


% Gradients
X_grad = (X * Theta' .* R - Y .* R) * Theta + lambda * X;
Theta_grad = (X *Theta' .* R - Y .* R)' * X + lambda * Theta; 
grad = [X_grad(:); Theta_grad(:)];

end

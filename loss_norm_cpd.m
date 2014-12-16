function [loss, gradloss] = loss_norm_cpd(x, U, params, enable_param_grad)
% Negative joint probability loss for linear Gaussian CPD
%
% params = [columns of U, sigma]; enable_param_grad = [(logical)*N, logical]

sigma = params(end);
mu = U * params(1:end-1)';
x_norm = (x - mu)/sigma;

logprobs = -log(sqrt(2*pi)) - log(sigma) - 1/2*(x_norm).^2;
loss = -sum(logprobs);

if nargout > 1,
  grad_w = repmat(x_norm, 1, size(U, 2)) .* U;
  
  grad_sigma = -1/sigma * (1 - x_norm.^2);

  gradloss = [-sum(grad_w, 1), -sum(grad_sigma)];
  if nargin >= 4,
    gradloss = gradloss(enable_param_grad);
  end
end

end

function [loss, gradloss] = loss_obs_dir_cpd(dx, obs_dir, params, enable_param_grad)
% Negative joint probability loss for directional observation CPD
%
% params = [kd, od, bd]; enable_param_grad = [logical, logical, logical]
%
% SAMPLE USAGE:
%{

dx = [0.25, 0.9, -0.11, -1, 0.2, -0.1];
obs_dir = [0, 1, -1, -1, 1, 0];
params = [10.124, 0.187, 0.2];
[loss, gradloss] = loss_obs_dir_cpd(dx, obs_dir, params)

%}

kd = params(1);
od = params(2);
bc = params(3);

idx_plus = (obs_dir > 0);
idx_minus = (obs_dir < 0);
idx_same = (obs_dir == 0);

sigmoid_plus = 1.0./(1.0 + exp(-kd*(dx-od)));
sigmoid_minus = 1.0./(1.0 + exp(-kd*(-dx-od)));

probs = zeros(size(dx));
probs(idx_plus) = sigmoid_plus(idx_plus) * (1-3*bc) + bc;
probs(idx_minus) = sigmoid_minus(idx_minus) * (1-3*bc) + bc;
probs(idx_same) = 1 - 2*bc - (1-3*bc) * (sigmoid_plus(idx_same) + sigmoid_minus(idx_same));
probs(probs<0) = 0; % Deals with numerical errors that bleed below 0
% NOTE: if any(probs == 0), then the loss will be inf!

loss = -sum(log(probs)); % loss is negative of joint data probability, prod [ prob(obs_dir | dx, params) ]

if nargout > 1,
  grad_kc_probs = zeros(size(dx));
  grad_kc_probs(idx_plus) = (1-3*bc) * sigmoid_plus(idx_plus) .* (1-sigmoid_plus(idx_plus)) .* (dx(idx_plus) - od);
  grad_kc_probs(idx_minus) = (1-3*bc) * sigmoid_minus(idx_minus) .* (1-sigmoid_minus(idx_minus)) .* (-dx(idx_minus) - od);
  grad_kc_probs(idx_same) = (3*bc-1) * ( sigmoid_plus(idx_same) .* (1-sigmoid_plus(idx_same)) .* (dx(idx_same) - od) + ...
    sigmoid_minus(idx_same) .* (1-sigmoid_minus(idx_same)) .* (-dx(idx_same) - od) );
  
  grad_oc_probs = zeros(size(dx));
  grad_oc_probs(idx_plus) = (1-3*bc) * sigmoid_plus(idx_plus) .* (1-sigmoid_plus(idx_plus)) * -kd;
  grad_oc_probs(idx_minus) = (1-3*bc) * sigmoid_minus(idx_minus) .* (1-sigmoid_minus(idx_minus)) * -kd;
  grad_oc_probs(idx_same) = (1-3*bc) * kd * ( sigmoid_plus(idx_same) .* (1-sigmoid_plus(idx_same)) + ...
    sigmoid_minus(idx_same) .* (1-sigmoid_minus(idx_same)) );
  
  grad_bc_probs = zeros(size(dx));
  grad_bc_probs(idx_plus) = 1 - 3*sigmoid_plus(idx_plus);
  grad_bc_probs(idx_minus) = 1 - 3*sigmoid_minus(idx_minus);
  grad_bc_probs(idx_same) = -2 + 3*(sigmoid_plus(idx_same) + sigmoid_minus(idx_same));
  
  gradloss = [ ...
    -sum(grad_kc_probs ./ probs), ...
    -sum(grad_oc_probs ./ probs), ...
    -sum(grad_bc_probs ./ probs)];
  if nargin >= 4,
    gradloss = gradloss(enable_param_grad);
  end
end

end

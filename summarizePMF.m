function [stats] = summarizePMF(pmfs, states, bin_widths, alpha)
% summarizePMF  Compute expected/mode/lower CI/upper CI for PMFs
%
% - pmfs: TxN matrix, where each row contains N NORMALIZED probability mass
%         distributions
% - states: 1xN vector containing centered values for each state bin
% - bin_widths: 1xN vector containing width of each state bin
% - alpha: (optional) alpha level, default: 0.05
%
% - stats: Tx4 matrix, corresponding to expected/mode/lower confidence
%          interval/upper confidence interval, for each row of pmfs

num_obs = size(pmfs, 1);
stats = zeros(num_obs, 4);

stats(:, 1) = sum(pmfs .* repmat(states, num_obs, 1), 2);
[~, mode_idxes] = max(pmfs, [], 2);
stats(:, 2) = states(mode_idxes);
cmfs = cumsum(pmfs, 2);

if nargin < 4 || isempty(alpha),
  alpha = 0.05;
end
if alpha > 0.5,
  alpha = 1 - alpha;
end
conf_int_low = alpha/2;
conf_int_high = 1 - conf_int_low;

for i = 1:num_obs,
  idx_upper_ci_low = find(cmfs(i, :) >= conf_int_low, 1, 'first');
  if isempty(idx_upper_ci_low),
    disp('Could not find CI low index!');
    keyboard;
  end
  prob_upper_ci_low = cmfs(i, idx_upper_ci_low);
  prob_lower_ci_low = 0;
  if idx_upper_ci_low > 1,
    prob_lower_ci_low = cmfs(i, idx_upper_ci_low - 1);
  end
  stats(i, 3) = states(idx_upper_ci_low) - ...
    bin_widths(idx_upper_ci_low)/2 + ...
    bin_widths(idx_upper_ci_low) * (conf_int_low - prob_lower_ci_low) / ...
    (prob_upper_ci_low - prob_lower_ci_low);
  
  idx_upper_ci_high = find(cmfs(i, :) >= conf_int_high, 1, 'first');
  if isempty(idx_upper_ci_high),
    disp('Could not find CI high index!');
    keyboard;
  end
  prob_upper_ci_high = cmfs(i, idx_upper_ci_high);
  prob_lower_ci_high = 0;
  if idx_upper_ci_high > 1,
    prob_lower_ci_high = cmfs(i, idx_upper_ci_high - 1);
  end
  stats(i, 4) = states(idx_upper_ci_high) - ...
    bin_widths(idx_upper_ci_high)/2 + ...
    bin_widths(idx_upper_ci_high) * (conf_int_high - prob_lower_ci_high) / ...
    (prob_upper_ci_high - prob_lower_ci_high);
end

end

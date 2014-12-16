function probs = boundedLinearGaussianCPD(x, mu, sigma, num_bins)
% Computes Prob(x|...) ~= * normpdf(x: mu, sigma)
%
%
% *: CPDs beyond [0, 1] spatial bounds are attributed to boundary states:
%    Prob(x=0|...) = normpdf(x=0: mean, stdev) +
%      (normcdf(x=0: mean, stdev) - 0)
%    Prob(x=1|...) = normpdf(x=1: mean, stdev) +
%      (1 - normcdf(x=0: mean, stdev))
%
% - num_bins: bin size for histogram approximation of cont. distribution
% - either size(mu) == 1, or size(mu) == size(x)
% - outputs valid PMFs for p(y=[0-1]|...) that sum to 1
% - assumes x has bounded range: [0, 1]

mu = min(max(mu, 0), 1); % Bound mean by range of y
probs_unnorm = normpdf(x, mu, sigma) / num_bins; % Divide by num_bins so that sum(probs_unnorm) ~= 1 (notwithstanding bounded range)

% Fold left- and right-extremities of Gaussian cumulative density
% function (CDF) onto boundary states
probs_unnorm(1, :) = probs_unnorm(1, :) + (normcdf(0, mu(1, :), sigma) - 0);
probs_unnorm(end, :) = probs_unnorm(end, :) + (1 - normcdf(1, mu(end, :), sigma));

% Normalize probability mass function (PMF)
norms = sum(probs_unnorm, 1);
norms(norms==0) = 1; % Ignore histogram degeneracy errors
norms = norms ./ num_bins; % for the underlying PDF to integrate to 1, the corresponding histogram bin center PMF must sum up to num_bins
probs = probs_unnorm ./ repmat(norms, num_bins, 1);

end

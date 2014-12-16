% Dynamic Bayesian Network (DBN) for tracking the pose of a toy robot:
% - 1-D bounded space: robot's pose x \in [0, 1]
% - holonomic motion: at each time step, robot can move <- 0 or -> 1; robot
%     remains at boundary if tries to move beyond boundary
% - heading control: user has control over robot's heading direction,
%     i.e. u \in {-1, 1}, but this signal is affected by the robot's
%     constant but unknown speed
%   - additive motion noise: robot's pose is perturbed by some random
%     noise (for simplicity, assume noise present even if stationary)
%   - drift bias: robot's pose may drift by a constant but unknown bias
%     (for simplicity, assume bias present even if stationary)
% - direction observations: external observer can sense robot's direction
%     of motion, d \in {-1, 0, 1}
%   - observation variability: probability of reporting directional motion
%     (i.e. direction == -1 | 1) <= 1; improves proportional to magnitude
%     of pose change
%   - observation tolerance: pose changes below a certain expected
%     magnitude are perceived as non-motion (i.e. direction == 0)
%   - observation error: with a certain probability, external observer
%     will report an arbitrary observed direction, which may or may not
%     be correct
% - pose observations: observer can sense robot's pose, f \in [0, 1]
%   - with additive noise (e.g. blurred vision)
%   - if robot is at boundary, assume that "would-be" noisy observations
%     beyond boundary are mapped to an observation "at" the boundary
%
% The corresponding DBN is shaped as follows (t: time step)
%
%          u[t]
%           |
%           v
% x[t-1]->x[t]
%     \   /  |
%       v    v
%      d[t] z[t]
%
%
% This DBN has a linear Gaussian propagation conditional probability
% distribution (CPD):
% Prob(x[t] | x[t-1], u[t]) ~= *
%   normpdf( x[t]: mean=x[t-1] + wx_bias + wx_spd * u[t], stdev=sx )
%
% *: CPDs beyond [0, 1] spatial bounds are attributed to boundary states:
%    Prob(x[t]=0|...) = normpdf(x[t]=0: mean, stdev) +
%      (normcdf(x[t]=0: mean, stdev) - 0)
%    Prob(x[t]=1|...) = normpdf(x[t]=1: mean, stdev) +
%      (1 - normcdf(x[t]=0: mean, stdev))
%
% The propagation CPD has the following model parameters:
% - wx_bias: drift bias
% - wx_spd: speed
% - sx: additive motion noise
%
%
% Likelihoods of directional observations are modeled as sigmoid CPDs:
% Prob(d[t]=+1 | x[t], x[t-1]) = bd +
%   (1-3*bd)*Sigmoid(kd( (x[t] - x[t-1]) - od))
% Prob(d[t]=-1 | x[t], x[t-1]) = bd +
%   (1-3*bd)*Sigmoid(kd(-(x[t] - x[t-1]) - od))
% Prob(d[t]=0|...) = 1 - Prob(d[t]=+1|...) - Prob(d[t]=-1|...)
%
% where Sigmoid(x) = 1/(1 + exp(-x))
%
% This directional observation CPD has the following model parameters:
% - kd: slope of sigmoid, reflecting observation variability
% - od: expected offset, reflecting observation tolerance
% - bd: baseline likelihood, reflecting observation error
%
%
% The CPD for pose observation is modeled as a linear Gaussian:
% Prob(z[t] | x[t]) = normpdf( x[t]: mean=z[t], stdev=sz )
%
% The pose observation CPD has the following model parameter:
% - sz: observation variability
classdef RobotModel < handle
  properties (SetAccess='protected')
    params % structure containing graphical model parameters
  end
  
  
  
  properties (Constant)
    optimize_prop_cpd_using_fmincon = false; % as opposed to linear least squares

    prop_optim_options = optimoptions('fmincon', ...
      'Algorithm', 'interior-point', ...
      'GradObj', 'on', 'Display', 'none');
    prop_optim_params_lb = [-1, -1, 0]; % Hand-chosen lower bounds for wx_bias, wx_spd, sx
    prop_optim_params_ub = [1, 1, inf]; % Hand-chosen upper bounds for wx_bias, wx_spd, sx
    
    obs_dir_optim_options = optimoptions('fmincon', ...
      'Algorithm', 'interior-point', ...
      'GradObj', 'on', 'Display', 'none');
    obs_dir_optim_params_lb = [1, 0, 0]; % Hand-chosen lower bounds for kd, od, bd
    obs_dir_optim_params_ub = [1e4, 1, 1.0/3]; % Hand-chosen upper bounds for kd, od, bd

    % Hand-chosen tolerances for model parameters
    eps_wx_bias = 1e-8;
    eps_wx_spd = 1e-8;
    eps_sx = 1e-10;
    eps_kd = 1e-9;
    eps_od = 1e-9;
    eps_bd = 1e-9;
    eps_sz = 1e-10;
  end
  
  
  
  methods(Static)
    % Vizualizes observed/controls data set and PGM inference outputs
    function plotHistogram(data_all, cache, map_states, filtered_probs, ...
        smoothed_probs, fig_id, alpha)
      if nargin < 6,
        fig_id = 0;
      end
      if nargin < 7,
        alpha = [];
      end
      show_map = false;
      show_filter = false;
      show_smooth = false;
      if nargin >= 3 && ~isempty(map_states),
        show_map = true;
      end
      if nargin >= 4 && ~isempty(filtered_probs),
        show_filter = true;
        filtered_stats = summarizePMF(filtered_probs./cache.num_bins, ...
          cache.x_vec, 1.0/cache.num_bins*ones(cache.num_bins, 1), alpha);
      end
      if nargin >= 5 && ~isempty(smoothed_probs),
        show_smooth = true;
        smoothed_stats = summarizePMF(smoothed_probs./cache.num_bins, ...
          cache.x_vec, 1.0/cache.num_bins*ones(cache.num_bins, 1), alpha);
      end

      time_indices = 1:length(data_all);
      time_indices_w_zero = 0:length(data_all);

      if fig_id > 0,
        figure(fig_id);
      end
      clf;
      
      subplot(3, 1, 1);
      hold on;
      
      if show_filter,
        h = errorbar(time_indices_w_zero, filtered_stats(:, 1), ...
          filtered_stats(:, 4) - filtered_stats(:, 1), ...
          filtered_stats(:, 1) - filtered_stats(:, 3));
        errorbar_tick(h, 0.4, 'units');
        set(h, 'Color', 'g');
      end
      
      if show_smooth,
        h = errorbar(time_indices_w_zero, smoothed_stats(:, 1), ...
          smoothed_stats(:, 4) - smoothed_stats(:, 1), ...
          smoothed_stats(:, 1) - smoothed_stats(:, 3));
        errorbar_tick(h, 0.4, 'units');
        set(h, 'Color', 'b');
      end
      
      if show_map,
        plot(time_indices_w_zero, map_states, '-k', 'LineWidth', 2);
      end
      
      valid_z_idx = arrayfun(@(s) ~isempty(s.z), data_all);
      if any(valid_z_idx),
        stem(time_indices(valid_z_idx), [data_all(valid_z_idx).z], 'or', 'LineWidth', 2);
      end
      hold off;
      xlabel('time (sec)     MAP (black -) | Exp(smooth) (blue -) | Exp(filter) (green -) | Observed Pose (red o)');
      ylabel('Latent state (robot pose)');
      ax = axis(); ax(1) = 0; ax(2) = time_indices(end); ax(3) = 0; ax(4) = 1; axis(ax);
      
      subplot(3, 1, 2);
      hold on;
      map_diff = [];
      diff_filter = [];
      diff_smooth = [];
      if show_map,
        map_diff = map_states(2:end) - map_states(1:end-1);
      end
      if show_filter,
        diff_filter = filtered_stats(2:end, 1) - filtered_stats(1:end-1, 1);
        plot(time_indices, diff_filter, '-vg');
      end
      if show_smooth,
        diff_smooth = smoothed_stats(2:end, 1) - smoothed_stats(1:end-1, 1);
        plot(time_indices, diff_smooth, '--^b');
      end
      if show_map,
        plot(time_indices, map_diff, ':xk');
      end
      max_abs_diff = max(abs([map_diff; diff_filter; diff_smooth]));
      if isempty(max_abs_diff) || max_abs_diff == 0,
        max_abs_diff = 1;
      end
      valid_d_idx = arrayfun(@(s) ~isempty(s.d), data_all);
      if any(valid_d_idx),
        stem(time_indices(valid_d_idx), max_abs_diff*[data_all(valid_d_idx).d], 'ro', 'LineWidth', 1.5);
      end;
      hold off;
      xlabel('time (sec)     Diff in: MAP (black -) | Exp(smooth) (blue -) | Exp(filter) (green -) | Observed Dir (red o)');
      ylabel('Change in latent state (robot pose)');
      ax = axis(); ax(1) = 0; ax(2) = time_indices(end); ax(3) = -max_abs_diff; ax(4) = max_abs_diff; axis(ax);
      
      active_u_idx = arrayfun(@(s) ~isempty(s.u), data_all);
      subplot(3, 1, 3);
      hold on;
      stem(time_indices(active_u_idx), 1*[data_all(active_u_idx).u], '-xb');
      hold off;
      xlabel('time (sec)     Control Input (blue x)');
      ax = axis(); ax(1) = 0; ax(2) = time_indices(end); ax(3) = -1; ax(4) = 1; axis(ax);
    end
    
    
    % Assesses whether 2 sets of parameters are sufficiently similar
    function same = compareParams(params_a, params_b, epsilon)
      if nargin < 3,
        epsilon = 0;
      end
      
      same = ...
        (abs(params_a.wx_bias - params_b.wx_bias) < epsilon*RobotModel.eps_wx_bias) && ...
        (abs(params_a.wx_spd - params_b.wx_spd) < epsilon*RobotModel.eps_wx_spd) && ...
        (abs(params_a.sx - params_b.sx) < epsilon*RobotModel.eps_sx) && ...
        (abs(params_a.kd - params_b.kd) < epsilon*RobotModel.eps_kd) && ...
        (abs(params_a.od - params_b.od) < epsilon*RobotModel.eps_od) && ...
        (abs(params_a.bd - params_b.bd) < epsilon*RobotModel.eps_bd) && ...
        (abs(params_a.sz - params_b.sz) < epsilon*RobotModel.eps_sz);
    end % function compareParams()
    
    
    % Optimizes model parameters given temporal batch of observation data,
    % and (estimated) sequence of robot poses
    function params_new = optimizeParams(data_all, x_states, params_old)
      if size(x_states, 2) > 1, % Generalized (a.k.a. soft-assignment) EM
        error('RobotModel:SoftEM', 'Soft EM has not been implemented');
        
      else % size(states, 2) == 1, i.e. Hard-assignment EM
        params_new = params_old;
        num_time_steps = numel(data_all);
        if num_time_steps == 0,
          error('RobotModel:EmptyDataset', 'data_all has 0 samples');
        end

        x_past = x_states(1:end-1);
        x_curr = x_states(2:end);
        x_diff = x_curr - x_past;
        
        % Optimize wx_bias, wx_spd (,sx):
        % either via constrained minimization of negative log joint prob,
        % or via linear least squares
        u_curr = nan*zeros(size(data_all, 1), 1);
        valid_u_idx = arrayfun(@(d) ~isempty(d.u), data_all);
        u_curr(valid_u_idx) = [data_all(valid_u_idx).u]';
        
        At = [ones(sum(valid_u_idx), 1), u_curr(valid_u_idx)];
        if cond(At) > 10000, % Hand-chosen singularity pre-check
          warning('RobotModel:ParamOptimFailed', ...
            'Cannot optimize params for propagate() since design matrix A is near-singular');
        else
          if RobotModel.optimize_prop_cpd_using_fmincon,
            % Optimize using MATLAB's generalized non-linear function minimization
            prop_optim_loss = @(params) loss_norm_cpd(x_diff(valid_u_idx), At, params);
            [prop_optim_newparams, ~, prop_optim_flag] = ...
              fmincon(prop_optim_loss, ...
              [params_old.wx_bias, params_old.wx_spd, params_old.sx], ...
              [], [], [], [], ...
              RobotModel.prop_optim_params_lb, RobotModel.prop_optim_params_ub, ...
              [], RobotModel.prop_optim_options);
            if prop_optim_flag < 0,
              warning('RobotModel:ParamOptimFailed', ...
                'Bounded optimization of propagate() params failed');
            else
              params_new.wx_bias = prop_optim_newparams(1);
              params_new.wx_spd = prop_optim_newparams(2);
              params_new.sx = prop_optim_newparams(3);
            end

          else % Optimize using linear least squares (LLS)
            % Assume that LLS can be used to optimize bounded linear
            % Gaussian CPD's params, akin to its unbounded variant

            prop_optim_newparams = At\x_diff(valid_u_idx);
            params_new.wx_bias = prop_optim_newparams(1);
            params_new.wx_spd = prop_optim_newparams(2);
          end
        end
        
        % Optimize kd, od, bd via constrained minimization of negative
        % log joint prob
        valid_d_idx = arrayfun(@(d) ~isempty(d.d), data_all);
        dx_instances = x_diff(valid_d_idx);
        obs_dir_instances = [data_all(valid_d_idx).d]';
        obs_dir_optim_loss = @(params) loss_obs_dir_cpd(dx_instances, obs_dir_instances, params);
        [obs_dir_optim_newparams, ~, obs_dir_optim_flag] = ...
          fmincon(obs_dir_optim_loss, ...
          [params_old.kd, params_old.od, params_old.bd], ...
          [], [], [], [], ...
          RobotModel.obs_dir_optim_params_lb, RobotModel.obs_dir_optim_params_ub, ...
          [], RobotModel.obs_dir_optim_options);
        if obs_dir_optim_flag < 0,
          warning('Model:ParamOptimFailed', ...
            'Bounded optimization of observe_dir() params failed');
        else
          params_new.kd = obs_dir_optim_newparams(1);
          params_new.od = obs_dir_optim_newparams(2);
          params_new.bd = obs_dir_optim_newparams(3);
        end
        
        % NOTE: should not optimize sz based on empirical estimates, since
        %       it would likely shrink true observation variability if
        %       estimated repeatedly
      end % Soft-/Hard-assignment EM
    end % function optimizeParams()
  end % methods(Static)
  
  
  
  methods
    % Creates new robot model PGM object
    function obj=RobotModel(params)
      obj.params = params;
    end

    
    % Update parameters of robot model PGM
    function updateParams(obj, new_params)
      obj.params = new_params;
    end

    
    % Propagates belief on latent state to next time step
    function probs = propagate(obj, cache, data_curr, data_past) %#ok<INUSD>
      % Expect cache = struct with fields: x_curr, x_past
      % Expect data_curr, data_past = structs with fields: u
      
      % Return identify mapping if no propagation observables
      if ~isfield(data_curr, 'x') || isempty(data_curr.x),
        probs = cache.eye_num_bins;
        return;
      end
      
      % Compute linear Gaussian conditional probability distribution (CPD)
      mu = cache.x_past + obj.params.wx_bias + obj.params.wx_spd*data_curr.u;
      probs = boundedLinearGaussianCPD(cache.x_curr, mu, obj.params.sx, ...
        cache.num_bins);
    end % function propagate()
    
    
    % Applies evidence to update belief on latent state
    function probs = observe(obj, cache, data_curr, data_past) %#ok<INUSD>
      % Expect cache = struct with fields: x_curr, x_past
      % Expect data_curr = struct with optional fields: d, z
      
      probs = ones(size(cache.x_curr));
      
      if isfield(data_curr, 'd') && ~isempty(data_curr.d),
        probs = probs .* sigmoidCPD( ...
          cache.x_past, cache.x_curr, data_curr.d, ...
          obj.params.kd, obj.params.od, obj.params.bd);
      end
      
      if isfield(data_curr, 'z') && ~isempty(data_curr.z),
        probs = probs .* boundedLinearGaussianCPD( ...
          cache.x_curr, data_curr.z, obj.params.sz, size(cache.x_curr, 1));
      end
    end % function observe()
  end % methods
end % classdef RobotModel

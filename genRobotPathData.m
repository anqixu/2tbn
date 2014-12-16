function ds = genRobotPathData(num_time_steps, init_pos, ...
  prop_behavior, prop_step_size, prop_bias, prop_stdev, ...
  observe_pose_stdev, observe_pose_freq, observe_dir_od, ...
  observe_dir_kd, observe_dir_bd, observe_dir_freq, sweep_tight_lower_cap)
% Generates time-series data for toy robot example
%
% ds.params: parameters of robot controller / dataset generator
% ds.data_all: struct array, containing the following fields:
%   - x \in [0, 1]: ground truth for pose of robot
%   - u \in {-1, 1}: steering direction (control input)
%   - d \in {-1, 0, 1}: observed direction of motion
%   - z \in [0, 1]: (noisy) observed pose of robot

% Set default values for optional arguments
if nargin < 8,
  observe_pose_freq = 1;
end
if nargin < 9,
  observe_dir_od = 0.1;
end
if nargin < 10,
  observe_dir_kd = 1000;
end
if nargin < 11,
  observe_dir_bd = 0;
end
if nargin < 12,
  observe_dir_freq = inf;
end
if nargin < 13,
  sweep_tight_lower_cap = 0.4;
end

% Parse arguments
sweep_tight_higher_cap = 1.0 - sweep_tight_lower_cap;
if ~strcmp(prop_behavior, 'stationary') && ...
    ~strcmp(prop_behavior, 'brownian') && ...
    ~strcmp(prop_behavior, 'sweep') && ...
    ~strcmp(prop_behavior, 'sweep_tight'),
  warning('genRobotPathData:CodingError', ...
    'Specified behavior [%s] is not recognized, setting to stationary', ...
    prop_behavior);
  prop_behavior = 'stationary';
end

ds.params.num_time_steps = num_time_steps;
ds.params.init_pos = min(max(init_pos, 0.0), 1.0);
ds.params.propagate_behavior = prop_behavior;
ds.params.propagate_step_size = prop_step_size;
ds.params.propagate_bias = prop_bias;
ds.params.propagate_stdev = prop_stdev;
ds.params.observe_pose_stdev = observe_pose_stdev;
ds.params.observe_pose_freq = observe_pose_freq;
ds.params.observe_dir_od = observe_dir_od;
ds.params.observe_dir_kd = observe_dir_kd;
ds.params.observe_dir_bd = observe_dir_bd;
ds.params.observe_dir_freq = observe_dir_freq;
ds.data_all = repmat(struct('x', [], 'u', [], 'd', [], 'z', []), ...
  ds.params.num_time_steps, 1);

pos = init_pos;
steering_dir = 1;
for i = 1:ds.params.num_time_steps,
  % Propagate
  if strcmp(prop_behavior, 'stationary'),
    u = 0;
  elseif strcmp(prop_behavior, 'brownian'),
    u = randi(3)-2;
  elseif strcmp(prop_behavior, 'sweep'),
    u = steering_dir;
  elseif strcmp(prop_behavior, 'sweep_tight'),
    u = steering_dir;
  end

  % Update pose
  prev_pos = pos;
  pos = pos + u*prop_step_size + normrnd(prop_bias, prop_stdev);
  if strcmp(prop_behavior, 'sweep_tight'),
    if pos >= sweep_tight_higher_cap,
      pos = sweep_tight_higher_cap;
      steering_dir = -1;
    elseif pos <= sweep_tight_lower_cap,
      pos = sweep_tight_lower_cap;
      steering_dir = 1;
    end
  else
    if pos >= 1,
      pos = 1;
      steering_dir = -1;
    elseif pos <= 0,
      pos = 0;
      steering_dir = 1;
    end
  end  
  
  % Observe direction
  if mod(i-1, ds.params.observe_dir_freq) == 0,
    prob_obs_pos_dir = sigmoidCPD(prev_pos, pos, 1, ...
      ds.params.observe_dir_kd, ds.params.observe_dir_od, ...
      ds.params.observe_dir_bd);
    prob_obs_neg_dir = sigmoidCPD(prev_pos, pos, -1, ...
      ds.params.observe_dir_kd, ds.params.observe_dir_od, ...
      ds.params.observe_dir_bd);
    r = rand();
    if r < prob_obs_pos_dir,
      d = +1;
    elseif r < prob_obs_pos_dir + prob_obs_neg_dir,
      d = -1;
    else
      d = 0;
    end
  else
    d = [];
  end

  % Observe pose
  if mod(i-1, ds.params.observe_pose_freq) == 0,
    z = normrnd(pos, observe_pose_stdev);
    if z > 1,
      z = 1;
    elseif z < 0,
      z = 0;
    end
  else
    z = [];
  end
  
  % Collect data
  ds.data_all(i).x = pos;
  ds.data_all(i).u = u;
  ds.data_all(i).d = d;
  ds.data_all(i).z = z;
end

end

% TEST_ROBOT.m

%% Set up environment
clear;
close all;
restoredefaultpath;

rng(1729);

%% Generate simulated observed data

num_time_steps = 201;
init_pos = 0.3;
%prop_behavior = 'stationary';
%prop_behavior = 'brownian';
prop_behavior = 'sweep';
%prop_behavior = 'sweep_tight';
prop_step_size = 0.01;
prop_bias = 0.0;
prop_stdev = 0.01;
observe_pose_stdev = 0.1;
observe_pose_freq = 20;
observe_dir_od = 0.02;
observe_dir_kd = 150;
observe_dir_bd = 0.1;
observe_dir_freq = 5;
sweep_tight_lower_cap = 0.2;

dataset = genRobotPathData(num_time_steps, init_pos, ...
  prop_behavior, prop_step_size, ...
  prop_bias, prop_stdev, observe_pose_stdev, observe_pose_freq, ...
  observe_dir_od, observe_dir_kd, observe_dir_bd, observe_dir_freq, ...
  sweep_tight_lower_cap);
ds = dataset.data_all;

%% Initialize engine

% Set to 0 to only run filtering+smoothing+MAP
num_em_steps = 5;

num_bins = 300;
hard_em_type = 'smooth';
em_params_eps_gain = 1;
params.wx_bias = 0.1;
params.wx_spd = 0.05;
params.sx = 0.05;
params.od = 0.5;
params.kd = 25;
params.bd = 0.2;
params.sz = 0.1;

model = RobotModel(params);
model_dup = RobotModel(params);
engine = HistoEngineEM(model, params, num_bins, [], ds, hard_em_type);
engine_dup = HistoEngineEM(model_dup, params, num_bins, [], ds, hard_em_type);

%% Run smoothing/filtering/MAP

inference_t = -1;
paramfit_t = -1;
ljp_t = -1;
if num_em_steps <= 0,
  tic;
  [smoothing_probs, filtering_probs] = engine.batchSmooth(ds, true);
  map_states = engine.extractMAP();
  inference_t = toc;

  % Run log joint prob
  tic;
  state_traj = map_states;
  logjointprob = engine.logJointProb(ds, state_traj);
  ljp_t = toc;

  % Run param fitting
  tic;
  params_new = model.optimizeParams(ds, state_traj, params);
  paramfit_t = toc;
  
  % Plot results
  model.plotHistogram(ds, engine.cache, state_traj, ...
    filtering_probs, smoothing_probs);
end;

%% Run EM
em_t = -1;
if num_em_steps > 0,
  engine.reset();
  tic;
  engine.runEM(num_em_steps, false, em_params_eps_gain, true);
  em_t = toc;
end

%% Report results
if num_em_steps > 0,
  state_vec_matrix = repmat(engine.cache.x_vec, size(ds, 1)+1, 1);
  
  init_params = engine.em.params_list{1};
  final_params = engine.em.params_list{end};

  engine_dup.reset();
  engine_dup.updateParams(init_params);
  cache = engine_dup.cache;
  cache.ds = dataset;
  [smoothing_probs_first, filtering_probs_first] = engine_dup.batchSmooth(ds, true);
  map_states_first = engine_dup.extractMAP();
  exp_smoothed_states_first = mean(smoothing_probs_first.*state_vec_matrix, 2);
  ljp_first = engine_dup.logJointProb(ds, exp_smoothed_states_first);
  model_dup.plotHistogram(ds, cache, map_states_first, ...
    filtering_probs_first, smoothing_probs_first, 1, []);
  title(sprintf('First EM iter (Log Joint Prob w/ Expected Smoothed States: %.4e)', ljp_first));

  engine_dup.reset();
  engine_dup.updateParams(final_params);
  cache = engine_dup.cache;
  cache.ds = dataset;
  [smoothing_probs_last, filtering_probs_last] = engine_dup.batchSmooth(ds, true);
  map_states_last = engine_dup.extractMAP();
  exp_smoothed_states_last = mean(smoothing_probs_last.*state_vec_matrix, 2);
  ljp_last = engine_dup.logJointProb(ds, exp_smoothed_states_last);
  model_dup.plotHistogram(ds, cache, map_states_last, ...
    filtering_probs_last, smoothing_probs_last, 2, []);
  title(sprintf('Final EM iter (Log Joint Prob w/ Expected Smoothed States: %.4e)', ljp_last));

  curr_params = init_params;
  fprintf('\n');
  fprintf('First Iter Params (vs GT | diff):\n');
  fprintf('- wx_bias: %.4f (%.4f | %.4f)\n', curr_params.wx_bias, prop_bias, abs(curr_params.wx_bias - prop_bias));
  fprintf('- wx_spd: %.4f (%.4f | %.4f)\n', curr_params.wx_spd, prop_step_size, abs(curr_params.wx_spd - prop_step_size));
  fprintf('- sx: %.4f (%.4f | %.4f)\n', curr_params.sx, prop_stdev, abs(curr_params.sx - prop_stdev));
  fprintf('- od: %.4f (%.4f | %.4f)\n', curr_params.od, observe_dir_od, abs(curr_params.od - observe_dir_od));
  fprintf('- kd: %.4f (%.4f | %.4f)\n', curr_params.kd, observe_dir_kd, abs(curr_params.kd - observe_dir_kd));
  fprintf('- bd: %.4f (%.4f | %.4f)\n', curr_params.bd, observe_dir_bd, abs(curr_params.bd - observe_dir_bd));
  fprintf('- sz: %.4f (%.4f | %.4f)\n', curr_params.sz, observe_pose_stdev, abs(curr_params.sz - observe_pose_stdev));
  fprintf('\n');

  curr_params = final_params;
  fprintf('Last Iter Params (vs GT | diff):\n');
  fprintf('- wx_bias: %.4f (%.4f | %.4f)\n', curr_params.wx_bias, prop_bias, abs(curr_params.wx_bias - prop_bias));
  fprintf('- wx_spd: %.4f (%.4f | %.4f)\n', curr_params.wx_spd, prop_step_size, abs(curr_params.wx_spd - prop_step_size));
  fprintf('- sx: %.4f (%.4f | %.4f)\n', curr_params.sx, prop_stdev, abs(curr_params.sx - prop_stdev));
  fprintf('- oc: %.4f (%.4f | %.4f)\n', curr_params.od, observe_dir_od, abs(curr_params.od - observe_dir_od));
  fprintf('- kc: %.4f (%.4f | %.4f)\n', curr_params.kd, observe_dir_kd, abs(curr_params.kd - observe_dir_kd));
  fprintf('- bc: %.4f (%.4f | %.4f)\n', curr_params.bd, observe_dir_bd, abs(curr_params.bd - observe_dir_bd));
  fprintf('- sz: %.4f (%.4f | %.4f)\n', curr_params.sz, observe_pose_stdev, abs(curr_params.sz - observe_pose_stdev));
  fprintf('\n');  
end

fprintf('TEST_ROBOT.m\n- inference: %.4f sec\n- log joint prob: %.4f sec\n- param fit: %.4f sec\n- em: %.4f sec\n\n', ...
  inference_t, ljp_t, paramfit_t, em_t);

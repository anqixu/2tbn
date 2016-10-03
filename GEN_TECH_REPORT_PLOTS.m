% GEN_TECH_REPORT_PLOTS.m

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
%prop_behavior = 'sweep';
prop_behavior = 'sweep_tight';
prop_step_size = 0.01;
prop_bias = 0.0;
prop_stdev = 0.01;
observe_pose_stdev = 0.1;
observe_pose_freq = 15;
observe_dir_od = 0.02;
observe_dir_kd = 150;
observe_dir_bd = 0.1;
observe_dir_freq = 5;
sweep_tight_lower_cap = 0.3;

dataset = genRobotPathData(num_time_steps, init_pos, ...
  prop_behavior, prop_step_size, ...
  prop_bias, prop_stdev, observe_pose_stdev, observe_pose_freq, ...
  observe_dir_od, observe_dir_kd, observe_dir_bd, observe_dir_freq, ...
  sweep_tight_lower_cap);
ds = dataset.data_all;

%% Initialize engine and run filtering through first N time steps

N_pre_filter_steps = 45;
num_bins = 100;
hard_em_type = 'smooth';
em_params_eps_gain = 1;
params.wx_bias = 0.05;
params.wx_spd = 0.15;
params.sx = 0.05;
params.od = 0.3;
params.kd = 25;
params.bd = 0.1;
params.sz = 0.1;

model = RobotModel(params);
engine = HistoEngineEM(model, params, num_bins, [], ds(1:N_pre_filter_steps), hard_em_type);
pre_filtered_pmfs = engine.batchFilter(ds(1:N_pre_filter_steps), false);

%% Plot propagation matrices for different values of u

data_curr = ds(N_pre_filter_steps+1);
data_past = ds(N_pre_filter_steps);
propCPDs = cell(3, 1);
data_curr.u = -1; propCPDs{1} = model.propagate(engine.cache, data_curr, data_past);
data_curr.u = 0; propCPDs{2} = model.propagate(engine.cache, data_curr, data_past);
data_curr.u = 1; propCPDs{3} = model.propagate(engine.cache, data_curr, data_past);
propCPDTitles = {'u = -1', 'u = 0', 'u = +1'};
probCPDFNames = {'prop_cpd_neg', 'prop_cpd_same', 'prop_cpd_pos'};

for i = 1:length(propCPDs),
  figure(0 + i);
  clf;
  set(gcf, 'Position', [300, 300, 225, 200]);
  hold on;
  max_prob_cap = max(propCPDs{i}(:))/15;
  imagesc([0, 1], [0, 1], propCPDs{i}, [0, max_prob_cap]);
  axis square;
  hbar = colorbar;
  hold off;
  box on;

  colormap('hot');
  hbarlab = ylabel(hbar, 'Prob(x_k|x_{k-1}, u_k)');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  title(propCPDTitles{i});
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.21, 0.22, 0.6, 0.625];
  set(gca, 'Position', hplotpos);
  hbarpos = [0.83, 0.22, 0.06, 0.625];
  set(hbar, 'YTick', [], 'Position', hbarpos);
  set(hbarlab, 'Position', [2.5, max_prob_cap/2, 1]);
  
  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.emf', probCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
  
  figure(3 + i);
  clf;
  set(gcf, 'Position', [300, 300, 210, 200]);
  hold on;
  hplot = imagesc([0, 1], [0, 1], propCPDs{i}, [0, max_prob_cap]);
  axis square;
  hold off;
  box on;

  colormap('hot');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.25, 0.22, 0.71, 0.74];
  set(gca, 'Position', hplotpos);

  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.raw.emf', probCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
end

%% Plot direction observation matrices for different values of d

data_curr = ds(N_pre_filter_steps+1);
data_past = ds(N_pre_filter_steps);
obsDirCPDs = cell(3, 1);
data_curr.d = -1; data_curr.z = []; obsDirCPDs{1} = model.observe(engine.cache, data_curr, data_past);
data_curr.d =  0; data_curr.z = []; obsDirCPDs{2} = model.observe(engine.cache, data_curr, data_past);
data_curr.d =  1; data_curr.z = []; obsDirCPDs{3} = model.observe(engine.cache, data_curr, data_past);
obsDirCPDTitles = {'d = -1', 'd = 0', 'd = +1'};
obsDirCPDFNames = {'obs_dir_cpd_neg', 'obs_dir_cpd_same', 'obs_dir_cpd_pos'};

for i = 1:length(obsDirCPDs),
  figure(6 + i);
  clf;
  set(gcf, 'Position', [300, 300, 225, 200]);
  hold on;
  max_prob_cap = max(obsDirCPDs{i}(:))/1;
  imagesc([0, 1], [0, 1], obsDirCPDs{i}, [0, max_prob_cap]);
  axis square;
  hbar = colorbar;
  hold off;
  box on;

  colormap('hot');
  hbarlab = ylabel(hbar, 'Prob(d_k|x_{k-1}, x_k)');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  title(obsDirCPDTitles{i});
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.21, 0.22, 0.6, 0.625];
  set(gca, 'Position', hplotpos);
  hbarpos = [0.83, 0.22, 0.06, 0.625];
  set(hbar, 'YTick', [], 'Position', hbarpos);
  set(hbarlab, 'Position', [2.5, max_prob_cap/2, 1]);
  
  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.emf', obsDirCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
  
  figure(9 + i);
  clf;
  set(gcf, 'Position', [300, 300, 210, 200]);
  hold on;
  hplot = imagesc([0, 1], [0, 1], obsDirCPDs{i}, [0, max_prob_cap]);
  axis square;
  hold off;
  box on;

  colormap('hot');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.25, 0.22, 0.71, 0.74];
  set(gca, 'Position', hplotpos);

  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.raw.emf', obsDirCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
end

%% Plot pose observation matrices for different values of z

data_curr = ds(N_pre_filter_steps+1);
data_past = ds(N_pre_filter_steps);
obsPoseCPDs = cell(3, 1);
data_curr.d = []; data_curr.z = 0.1; obsPoseCPDs{1} = model.observe(engine.cache, data_curr, data_past);
data_curr.d = []; data_curr.z = 0.5; obsPoseCPDs{2} = model.observe(engine.cache, data_curr, data_past);
data_curr.d = []; data_curr.z = 0.9; obsPoseCPDs{3} = model.observe(engine.cache, data_curr, data_past);
obsPoseCPDTitles = {'z = 0.1', 'z = 0.5', 'z = 0.9'};
obsPoseCPDFNames = {'obs_pose_cpd_01', 'obs_pose_cpd_05', 'obs_pose_cpd_09'};

for i = 1:length(obsPoseCPDs),
  figure(12 + i);
  clf;
  set(gcf, 'Position', [300, 300, 225, 200]);
  hold on;
  max_prob_cap = max(obsPoseCPDs{i}(:))/2;
  imagesc([0, 1], [0, 1], obsPoseCPDs{i}, [0, max_prob_cap]);
  axis square;
  hbar = colorbar;
  hold off;
  box on;

  colormap('hot');
  hbarlab = ylabel(hbar, 'Prob(z_k|x_{k-1}, x_k)');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  title(obsPoseCPDTitles{i});
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.21, 0.22, 0.6, 0.625];
  set(gca, 'Position', hplotpos);
  hbarpos = [0.83, 0.22, 0.06, 0.625];
  set(hbar, 'YTick', [], 'Position', hbarpos);
  set(hbarlab, 'Position', [2.5, max_prob_cap/2, 1]);
  
  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.emf', obsPoseCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
  
  figure(15 + i);
  clf;
  set(gcf, 'Position', [300, 300, 210, 200]);
  hold on;
  hplot = imagesc([0, 1], [0, 1], obsPoseCPDs{i}, [0, max_prob_cap]);
  axis square;
  hold off;
  box on;

  colormap('hot');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.25, 0.22, 0.71, 0.74];
  set(gca, 'Position', hplotpos);

  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.raw.emf', obsPoseCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
end

%% Plot prior state matrix at N+1'th time step

priorCPD = pre_filtered_pmfs(end, :);
priorCPDMat = repmat(priorCPD, num_bins, 1);
max_prob_cap = max(priorCPDMat(:));
prior_fnames = {'filter_prior', 'filter_prior_repmat'};

for i=0:1,
  figure(19+i);
  clf;
  set(gcf, 'Position', [300, 300, 210, 200]);
  hold on;
  if i > 0,
    hplot = imagesc([0, 1], [0, 1], priorCPDMat, [0, max_prob_cap]);
  end
  plot(engine.cache.x_vec, priorCPD/20, '-w', 'LineWidth', 8);
  plot(engine.cache.x_vec, priorCPD/20, '-k', 'LineWidth', 4);
  hold off;
  axis square;
  box on;

  colormap('hot');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  if i > 0,
    ylabel('x_k');
  else
    ylabel('b_f(x_{k-1})');
  end
  title('Prior Filtered Belief');
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);
  if i == 0,
    set(gca, 'YTick', 0:1, 'YTickLabel', {'', ''}, 'Color', 'k');
  end

  hplotpos = [0.26, 0.22, 0.71, 0.74];
  set(gca, 'Position', hplotpos);
  if i > 1,
    hbarpos = [0.83, 0.17, 0.085, 0.725];
    set(hbar, 'YTick', [], 'Position', hbarpos);
    set(hbarlab, 'Position', [2.5, max_prob_cap/2, 1]);
  end

  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.emf', prior_fnames{i+1}));
  catch
    fprintf('Printing failed\n');
  end
end

%% Plot propagation and observation matrices for N+1'th time step

data_curr = ds(N_pre_filter_steps+1);
data_past = ds(N_pre_filter_steps);
filterCPDs = cell(3, 1);
filterCPDs{1} = model.propagate(engine.cache, data_curr, data_past);
data_curr.z = []; filterCPDs{2} = model.observe(engine.cache, data_curr, data_past);
data_curr = ds(N_pre_filter_steps+1);
data_curr.d = []; filterCPDs{3} = model.observe(engine.cache, data_curr, data_past);
data_curr = ds(N_pre_filter_steps+1);
filterCPDTitles = {sprintf('u = %d', data_curr.u), sprintf('d = %d', data_curr.d), sprintf('z = %.1f', data_curr.z)};
filterCPDFNames = {sprintf('filter_prop_u%d', data_curr.u), sprintf('filter_obs_dir_d%d', data_curr.d), sprintf('filter_obs_pose_z%.1f', data_curr.z)};

for i = 1:length(filterCPDs),
  figure(20 + i);
  clf;
  set(gcf, 'Position', [300, 300, 210, 200]);
  hold on;
  hplot = imagesc([0, 1], [0, 1], filterCPDs{i}, [0, max_prob_cap]);
  axis square;
  hold off;
  box on;

  colormap('hot');

  axis([0, 1, 0, 1]);
  xlabel('x_{k-1}');
  ylabel('x_k');
  title(filterCPDTitles{i});
  set(gca, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);

  hplotpos = [0.25, 0.22, 0.71, 0.74];
  set(gca, 'Position', hplotpos);

  set(gcf,'paperpositionmode','auto');
  try
    print(gcf, '-dmeta', '-painters', sprintf('figures/%s.emf', filterCPDFNames{i}));
  catch
    fprintf('Printing failed\n');
  end
end

%% Plot joint belief, and filtered state matrix at N+1'th time step

%% Plot smooth-prior state matrix at N'th time step

%% Plot smoothed state matrix at N'th time step

%% TODO: prediction illustrations

%% TODO: MAP inference limitation illustrations

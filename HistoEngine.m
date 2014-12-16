classdef HistoEngine < handle
  properties (SetAccess='protected')
    prior_x_pmf % user-specified prior PMF at time==0; []: uniform prior
    
    model % object handle to graphical model
    
    settings
    % CONTENTS: (local version of cache)
    % settings.num_bins
    
    state
    % CONTENTS:
    % state.curr_time
    % state.x_latest_pmf % latest filtering === smoothing PMF
    %   (i.e. equivalent to state.x_filtered_pmfs(end, :))
    %   NOTE: these PMFs are used to approximate histogram PDFs, so they
    %         must sum up to num_bins for the corresponding histogram PDFs
    %         to integrate to 1
    % state.latest_max_x_logpmf % row vector containing
    %   max_{x_0:k-1}(factors), for MAP inference
    % state.x_filtered_pmfs % rows of histogram probabilities for different
    %   times
    % state.max_x_idxs % row vector containing indices for x_past at each
    %   time index
  end


  properties (Transient)
    cache
    % CONTENTS:
    % cache.num_bins
    %
    % cache.x_curr
    % cache.x_past
    % cache.x_vec
    %
    % cache.eye_num_bins
  end  
  
  
  properties
  end
  
  
  
  methods(Static)
    % Plots filtering+smoothing+MAP inference on observed data seq.
    function [engine, filtered_pmfs, smoothed_pmfs, map_states] = ...
        runModel(model, params, num_bins, data_all, ...
        prior_x_pmf, fig_id, conf_int_alpha)
      if nargin < 5,
        prior_x_pmf = [];
      end
      if nargin < 6,
        fig_id = 1;
      end
      if nargin < 7,
        conf_int_alpha = [];
      end
      engine = HistoEngine(model, params, num_bins, prior_x_pmf);
      [smoothed_pmfs, filtered_pmfs] = engine.batchSmooth(data_all, false, false);
      map_states = engine.extractMAP();
      model.plotHistogram(data_all, engine.cache, map_states, ...
        filtered_pmfs, smoothed_pmfs, fig_id, conf_int_alpha);
    end
    
    
    % Convenience fn: plots model output on training set
    function [engine, filtered_pmfs, smoothed_pmfs, map_states] = ...
        showTrainedModel(trial_obj, fig_id, conf_int_alpha)
      if nargin < 2,
        fig_id = 1;
      end
      if nargin < 3,
        conf_int_alpha = [];
      end
      
      params = trial_obj.train.opt_params;
      trained_model = RobotModel(params);
      num_bins = trial_obj.settings.num_bins;
      data_all = trial_obj.data.ds_train;
      trained_prior_x_pmf = [];
      
      [engine, filtered_pmfs, smoothed_pmfs, map_states] = ...
        HistoEngine.runModel(trained_model, params, num_bins, ...
        data_all, trained_prior_x_pmf, fig_id, conf_int_alpha);
    end
    
    
    % Convenience fn: plots model output on test set
    function [engine, filtered_pmfs, smoothed_pmfs, map_states] = ...
        showrunModel(trial_obj, fig_id, conf_int_alpha)
      if nargin < 2,
        fig_id = 1;
      end
      if nargin < 3,
        conf_int_alpha = [];
      end
      
      params = trial_obj.test.params;
      test_model = RobotModel(params);
      num_bins = trial_obj.settings.num_bins;
      data_all = trial_obj.data.ds_test;
      test_prior_x_pmf = trial_obj.test.prior_x_pmf;
      
      [engine, filtered_pmfs, smoothed_pmfs, map_states] = ...
        HistoEngine.runModel(test_model, params, num_bins, ...
        data_all, test_prior_x_pmf, fig_id, conf_int_alpha);
    end
  end
  
  
  
  methods
    % Initializes histogram inference engine object
    function obj=HistoEngine(model, params, num_bins, prior_x_pmf)
      if nargin < 4,
        obj.prior_x_pmf = [];
      else
        obj.prior_x_pmf = prior_x_pmf;
      end
        
      obj.model = model;
      obj.settings.num_bins = num_bins;
      obj.buildCache();
      obj.updateParams(params);
    end
    
    
    % (Re-)builds temporary cache data (useful for saving/loading objects)
    function buildCache(obj)
      obj.cache.num_bins = obj.settings.num_bins;
      obj.cache.bin_width = (1.0 - 0.0)/obj.settings.num_bins;
      obj.cache.eye_num_bins = eye(obj.settings.num_bins);
      
      obj.cache.x_vec = obj.cache.bin_width/2 + ...
        (0:(obj.settings.num_bins-1))*obj.cache.bin_width;
      obj.cache.x_past = repmat(obj.cache.x_vec, obj.settings.num_bins, ...
        1); % Each column is a different value for x for the past time index
      obj.cache.x_curr = obj.cache.x_past'; % each row is a different value
      % for x for the current time index
    end
    
   
    % Initiates prior belief on latent state
    function updatePriorPMF(obj, new_prior_x_pmf)
      obj.prior_x_pmf = new_prior_x_pmf;
      obj.resetState();
    end

    
    % Updates parameters for PGM
    function updateParams(obj, new_params)
      obj.model.updateParams(new_params);
      obj.resetState();
    end

    
    % Restores default state of histogram engine and removes any
    % previous inference results
    function resetState(obj)
      if isempty(obj.cache),
        obj.buildCache();
      end
      
      obj.state.curr_time = 0;
      
      % Attempt to apply user-specified prior_x_pmf
      obj.state.x_latest_pmf = [];
      if ~isempty(obj.prior_x_pmf),
        if size(obj.prior_x_pmf, 2) ~= obj.cache.num_bins,
          error('HistoEngine:InvalidPriorPMF', ...
            'User-specified prior PMF does not have the correct number of bins: found %d, expecting %d', ...
            size(obj.prior_x_pmf, 2), obj.cache.num_bins);
        else
          pmf_norm = sum(obj.prior_x_pmf);
          if pmf_norm == 0,
            error('HistoEngine:InvalidPriorPMF', ...
              'User-specified prior PMF sums to zero');
          elseif abs(pmf_norm - obj.cache.num_bins) > 1e-10,
            warning('HistoEngine:UnnormalizedPriorPMF', ...
              'User-specified prior PMF does not add up to num_bins, i.e. is a valid histogram PMF: found %.4f, expecting %d', ...
              sum(obj.prior_x_pmf), obj.cache.num_bins);
            obj.prior_x_pmf = obj.prior_x_pmf / pmf_norm * obj.cache.num_bins;
          end
          obj.state.x_latest_pmf = obj.prior_x_pmf;
        end
      end
      if isempty(obj.state.x_latest_pmf), % use uniform prior
        % NOTE: these PMFs are used to approximate histogram PDFs, so they
        %       must sum up to num_bins, so that the corresponding
        %       histogram PDFs integrate to 1
        obj.state.x_latest_pmf = ones(1, obj.cache.num_bins); 
      end
      
      obj.state.x_filtered_pmfs = obj.state.x_latest_pmf;
      obj.state.latest_max_x_logpmf = log(obj.state.x_latest_pmf);
      obj.state.max_x_idxs = ones(size(obj.state.x_latest_pmf))*-1;
      % NOTE: the first row is invalid, since they represent index of x_-1
      %       given choice of x_0
    end
    
    
    % Iteratively updates the filtering belief (propagation + observation)
    function stepFilter(obj, data_curr, data_past)
      % Setup local temporal relationships and increment time step
      if obj.state.curr_time <= 0,
        data_past = struct();
      end
      obj.state.curr_time = obj.state.curr_time + 1;

      % Compute posterior at current time step, given prior, observables for
      % propagation likelihoods, and observables for observation likelihoods
      logprobs_mat_prior = repmat(log(obj.state.x_latest_pmf), obj.cache.num_bins, 1);
      logprobs_mat_propagate = log(obj.model.propagate(obj.cache, data_curr, data_past));
      logprobs_mat_observe = log(obj.model.observe(obj.cache, data_curr, true));
      logprobs_mat_local_factors = logprobs_mat_propagate + logprobs_mat_observe;
      prob_mat_posterior = exp(logprobs_mat_prior + logprobs_mat_local_factors);
      % NOTE: columns of prob_mat_posterior do not need to sum up to
      %       num_bins since they are joint probabilities for observables
      %       of current time step and the current and past states
      
      % Collapse columns to marginalize out state for previous time step,
      % and compute normalized PMF of current time step
      probs_posterior = sum(prob_mat_posterior, 2)';
      norm_posterior = sum(probs_posterior);
      if norm_posterior == 0,
        error('HistoEngine:pmfZero', ...
          'Filtering failed: sum p(t_%d | obs_1:%d) == 0!', ...
          obj.state.curr_time, obj.state.curr_time);
      end
      norm_posterior = norm_posterior / obj.cache.num_bins;
      % NOTE: posterior PMFs must sum up to num_bins, rather than 1, so
      %       that their corresponding histogram PDFs integrate to 1
      obj.state.x_latest_pmf = probs_posterior ./ norm_posterior;
      obj.state.x_filtered_pmfs(obj.state.curr_time+1, :) = obj.state.x_latest_pmf;
      if any(isnan(obj.state.x_latest_pmf)),
        error('HistoEngine:pmfZero', ...
          'Filtering failed: p(t_%d | obs_1:%d) has NaN term!', ...
          obj.state.curr_time, obj.state.curr_time);
      end

      % Compute the maximal and argmax values over x_past of the max prior
      % factor and the local factor likelihoods, for MAP inference
      logprobs_mat_map_prior = repmat(obj.state.latest_max_x_logpmf, obj.cache.num_bins, 1);
      logprobs_mat_map_posterior = logprobs_mat_map_prior + logprobs_mat_local_factors;
      [obj.state.latest_max_x_logpmf, latest_max_x_logpmf_idxs] = max(logprobs_mat_map_posterior, [], 2);
      ambiguous_map_idx = max(logprobs_mat_map_posterior, [], 2) == min(logprobs_mat_map_posterior, [], 2);
      if any(ambiguous_map_idx),
        no_change_idxs = 1:obj.cache.num_bins;
        latest_max_x_logpmf_idxs(ambiguous_map_idx) = no_change_idxs(ambiguous_map_idx);
      end
      obj.state.latest_max_x_logpmf = obj.state.latest_max_x_logpmf';
      obj.state.max_x_idxs(obj.state.curr_time+1, :) = latest_max_x_logpmf_idxs';
    end
    
    
    % TODO: strip out model-dependent code from batchFilterPredict
    %{
    function [predicted_pmfs, pred_vals] = ...
        batchFilterPredict(obj, data_all, prediction_var, ...
        apply_pred_obs_after, predict_only_when_have_obs, ...
        verbose, pred_hist_num_bins)
      if nargin < 4,
        apply_pred_obs_after = false;
      end
      if nargin < 5,
        predict_only_when_have_obs = true;
      end
      if nargin < 6,
        verbose = false;
      end
      if nargin < 7,
        pred_hist_num_bins = 100;
      end

      num_time_steps = size(data_all, 1);

      if strcmp(prediction_var, 'i'),
        pred_vals = [0, 1];
        predicted_pmfs = nan(num_time_steps, 2);
      elseif strcmp(prediction_var, 'c'),
        pred_vals = [-1, 0, 1];
        predicted_pmfs = nan(num_time_steps, 3);
      elseif strcmp(prediction_var, 'f'),
        predicted_pmfs = nan(num_time_steps, pred_hist_num_bins);
        pred_vals = linspace(0, 1, pred_hist_num_bins+1) + 1/2/pred_hist_num_bins;
        pred_vals = pred_vals(1:end-1);
      else
        error('HistoEngine:BadPredictionVar', 'Invalid prediction var (%s): expecting {i, c, f}', prediction_var);
      end
      
      obj.resetState(false);
      
      % Apply forward sequential filtering
      data_past = struct();
      for time = 1:num_time_steps,
        if verbose,
          fprintf('Predicting %s: %3d / %3d steps...\n', prediction_var, time, num_time_steps);
        end
        data_curr = data_all(time);
        data_curr_full = data_curr;
        obj.state.curr_time = obj.state.curr_time + 1;
        
        % Remove prediction var if it exists in data_curr
        prediction_var_removed = false;
        if ~isempty(data_curr.(prediction_var)),
          data_curr.(prediction_var) = [];
          prediction_var_removed = true;
        end
          
        % Compute posterior at current time step, given prior, observables for
        % propagation likelihoods, and observables for observation likelihoods
        logprobs_mat_prior = repmat(log(obj.state.x_latest_pmf), obj.cache.num_bins, 1);
        logprobs_mat_propagate = obj.model.propagate_log(obj.cache, data_curr, data_past);
        logprobs_mat_observe_wo_pred = obj.model.observe_log(obj.cache, data_curr, true);
        prob_mat_posterior_wo_pred = exp(logprobs_mat_prior + logprobs_mat_propagate + logprobs_mat_observe_wo_pred);
        
        if prediction_var_removed || ~predict_only_when_have_obs,
          % Normalize POF matrix
          norm_mat_wo_pred = sum(prob_mat_posterior_wo_pred(:));
          if norm_mat_wo_pred == 0,
            error('HistoEngine:pmfZero', 'Filtered prediction failed: sum p(t_%d, t_%d | obs_1:%d) == 0!', obj.state.curr_time, obj.state.curr_time - 1, obj.state.curr_time);
          end
          log_prob_mat_posterior_wo_pred_norm = log(prob_mat_posterior_wo_pred) - log(norm_mat_wo_pred);
          if any(isnan(log_prob_mat_posterior_wo_pred_norm(:))),
            error('HistoEngine:pmfZero', 'Filtered prediction failed: p(t_%d, t_%d | obs_1:%d) has NaN term!', obj.state.curr_time, obj.state.curr_time - 1, obj.state.curr_time);
          end

          % Make prediction based on filtered posterior state
          if strcmp(prediction_var, 'i'),
            data_curr_pred = data_curr_full;
            data_curr_pred.c = [];
            data_curr_pred.f = [];

            data_curr_pred.i = 0;
            logprobs_mat_obs_only_pred = obj.model.observe_log(obj.cache, data_curr_pred, true);
            predicted_pmfs(time, 1) = sum(sum(exp(logprobs_mat_obs_only_pred + log_prob_mat_posterior_wo_pred_norm)));

            data_curr_pred.i = 1;
            logprobs_mat_obs_only_pred = obj.model.observe_log(obj.cache, data_curr_pred, true);
            predicted_pmfs(time, 2) = sum(sum(exp(logprobs_mat_obs_only_pred + log_prob_mat_posterior_wo_pred_norm)));

          elseif strcmp(prediction_var, 'c'),
            data_curr_pred = data_curr_full;
            data_curr_pred.i = [];
            data_curr_pred.f = [];

            data_curr_pred.c = -1;
            logprobs_mat_obs_only_pred = obj.model.observe_log(obj.cache, data_curr_pred, true);
            predicted_pmfs(time, 1) = sum(sum(exp(logprobs_mat_obs_only_pred + log_prob_mat_posterior_wo_pred_norm)));

            data_curr_pred.c = 0;
            logprobs_mat_obs_only_pred = obj.model.observe_log(obj.cache, data_curr_pred, true);
            predicted_pmfs(time, 2) = sum(sum(exp(logprobs_mat_obs_only_pred + log_prob_mat_posterior_wo_pred_norm)));

            data_curr_pred.c = 1;
            logprobs_mat_obs_only_pred = obj.model.observe_log(obj.cache, data_curr_pred, true);
            predicted_pmfs(time, 3) = sum(sum(exp(logprobs_mat_obs_only_pred + log_prob_mat_posterior_wo_pred_norm)));

          elseif strcmp(prediction_var, 'f'),
            data_curr_pred = data_curr_full;
            data_curr_pred.i = [];
            data_curr_pred.c = [];
            for i = 1:pred_hist_num_bins,
              data_curr_pred.f = pred_vals(i);
              logprobs_mat_obs_only_pred = obj.model.observe_log(obj.cache, data_curr_pred, true);
              predicted_pmfs(time, i) = sum(sum(exp(logprobs_mat_obs_only_pred + log_prob_mat_posterior_wo_pred_norm)));
            end

          else
            error('HistoEngine:BadPredictionVar', 'Invalid prediction var (%s): expecting {i, c, f}', prediction_var);
          end
        end
        
        % Re-run observation with non-omitted prediction var if necessary
        if prediction_var_removed && apply_pred_obs_after,
          data_curr = data_curr_full;
          logprobs_mat_observe = obj.model.observe_log(obj.cache, data_curr, true);
          prob_mat_posterior = exp(logprobs_mat_prior + logprobs_mat_propagate + logprobs_mat_observe);
        else
          prob_mat_posterior = prob_mat_posterior_wo_pred;
        end
        
        % Collapse columns to marginalize out state for previous time step,
        % and compute normalized PMF of current time step
        probs_posterior = sum(prob_mat_posterior, 2)';
        norm_posterior = sum(probs_posterior);
        if norm_posterior == 0,
          error('HistoEngine:pmfZero', 'Filtered post-prediction failed: sum p(t_%d | obs_1:%d) == 0!', obj.state.curr_time, obj.state.curr_time);
        end
        norm_posterior = norm_posterior / obj.cache.num_bins;
        obj.state.x_latest_pmf = probs_posterior ./ norm_posterior;
        obj.state.x_filtered_pmfs(obj.state.curr_time+1, :) = obj.state.x_latest_pmf;
        if any(isnan(obj.state.x_latest_pmf)),
          error('HistoEngine:pmfZero', 'Filtered post-prediction failed: p(t_%d | obs_1:%d) has NaN term!', obj.state.curr_time, obj.state.curr_time);
        end

        data_past = data_curr;
      end
      
      % Normalize prediction PMFs so that they sum up to 1
      norm_predicted_pmfs = sum(predicted_pmfs, 2);
      invalid_idx = isnan(norm_predicted_pmfs) | isinf(norm_predicted_pmfs);
      norm_predicted_pmfs(invalid_idx) = 1;
      predicted_pmfs = predicted_pmfs ./ repmat(norm_predicted_pmfs, 1, size(predicted_pmfs, 2));      
    end
    %}
    
    
    % Convenience fn: batch-computes filtered beliefs over latent states
    function filtered_pmfs = batchFilter(obj, data_all, verbose)
      if nargin < 3,
        verbose = false;
      end
      [~, filtered_pmfs] = obj.batchSmooth(data_all, verbose, true);
    end
    
    
    % Batch-computes filtered and smoothed beliefs over latent states
    function [smoothed_pmfs, filtered_pmfs] = batchSmooth(obj, ...
        data_all, verbose, filter_only)
      if nargin < 3,
        verbose = false;
      end
      if nargin < 4,
        filter_only = false;
      end
      
      num_time_steps = size(data_all, 1);
      obj.resetState();
      smoothed_pmfs = [];
      
      % Apply forward sequential filtering
      if verbose,
        for time = 1:num_time_steps,
          fprintf('Filtering+MAP %3d / %3d steps...\n', time, num_time_steps);
          if time == 1,
            obj.stepFilter(data_all(time));
          else
            obj.stepFilter(data_all(time), data_all(time-1));
          end
        end
      else
        for time = 1:num_time_steps,
          if time == 1,
            obj.stepFilter(data_all(time));
          else
            obj.stepFilter(data_all(time), data_all(time-1));
          end
        end
      end
    
      filtered_pmfs = obj.state.x_filtered_pmfs;
      
      % Apply backward sequential smoothing
      if ~filter_only,
        smoothed_pmfs = zeros(num_time_steps+1, obj.cache.num_bins);
        latest_smoothed_x_pmf = obj.state.x_latest_pmf;
        smoothed_pmfs(end, :) = latest_smoothed_x_pmf;
        
        for time = num_time_steps:-1:1,
          if verbose,
            fprintf('Smoothing %3d / %3d steps...\n', time, num_time_steps);
          end
          
          data_curr = data_all(time);
          if time > 1,
            data_past = data_all(time-1);
          else
            data_past = struct();
          end
          logprobs_mat_prior = repmat(log(obj.state.x_filtered_pmfs(time, :)), obj.cache.num_bins, 1);
          logprobs_mat_propagate = log(obj.model.propagate(obj.cache, data_curr, data_past));
          logprobs_mat_observe = log(obj.model.observe(obj.cache, data_curr, true));
          filtered_POF = exp(logprobs_mat_prior + logprobs_mat_propagate + logprobs_mat_observe);
          
          filtered_probs_posterior = sum(filtered_POF, 2);
          
          % filtered_POF = p(X_n, X_n+1, Z_n+1|U_1:n+1, Z_1:n),
          % and filtered_probs_posterior = p(X_n+1, Z_n+1:U_1:n+1, Z_1:n),
          % so if filtered_probs_posterior == 0 for some x_n+1 (and z_n+1),
          % then p(X_n, x_n+1, z_n+1|U_1:n+1, Z_1:n) would also have to be 0.
          %
          % We implement this by setting the corresponding entries of
          % log_smoothed_posterior_over_filtered_posterior to -inf,
          % and since its non-log, repmat form is multiplied with
          % filtered_POF, this will set the necessary rows of the product
          % probability to 0.
          log_smoothed_posterior_over_filtered_posterior = ...
            log(latest_smoothed_x_pmf') - log(filtered_probs_posterior);
          log_smoothed_posterior_over_filtered_posterior(filtered_probs_posterior==0) = -inf;
          smoothed_probs_backprior = exp(log(filtered_POF) + ...
            repmat(log_smoothed_posterior_over_filtered_posterior, 1, obj.cache.num_bins));
          
          latest_smoothed_x_pmf = sum(smoothed_probs_backprior, 1);
          norm_smoothed_x_pdf = sum(latest_smoothed_x_pmf);
          if norm_smoothed_x_pdf == 0,
            error('HistoEngine:pmfZero', ...
              'Smoothing failed: sum p(x_%d | obs_1:%d) == 0!', ...
              time, num_time_steps);
          end
          norm_smoothed_x_pdf = norm_smoothed_x_pdf ./ obj.cache.num_bins;
          % NOTE: smoothed PMFs must sum up to num_bins, rather than 1, so
          % that their corresponding histogram PDFs integrate to 1
          latest_smoothed_x_pmf = latest_smoothed_x_pmf ./ norm_smoothed_x_pdf;
          if any(isnan(latest_smoothed_x_pmf)),
            error('HistoEngine:pmfZero', ...
              'Smoothing failed: p(t_%d | obs_1:%d) has NaN term!', ...
              time, num_time_steps);
          end
          
          smoothed_pmfs(time, :) = latest_smoothed_x_pmf;
        end
      end
    end

    % Batch-computes most likely sequence of latent states (MAP inference)
    %
    % WARNING: must have called batchSmooth(), batchFilter(), or
    %          stepFilter() before
    function map_states = extractMAP(obj)
      if isempty(obj.cache),
        obj.buildCache();
      end
      
      if obj.state.curr_time <= 0,
        map_states = [];
      else
        % Perform backward trace: given X_T*, find X_(T-1)*, X_(T-2)*, etc.
        map_states = zeros(obj.state.curr_time+1, 1);
        [~, latest_map_idx] = max(obj.state.latest_max_x_logpmf);
        map_states(end) = obj.cache.x_vec(latest_map_idx);
        for time = obj.state.curr_time:-1:1,
          latest_map_idx = obj.state.max_x_idxs(time+1, latest_map_idx);
          map_states(time) = obj.cache.x_vec(latest_map_idx);
        end
      end
    end
    
    
    % Computes the log joint probability of an observed dataset and
    % provided latent states
    function logprobs = logJointProb(obj, data_all, states_all)
      % Return the log of the joint probability of the observed data and
      % a sequence of latent states
      %
      % states: (N+1)x1 column vector, representing latent state values at
      %   each of N time steps + prior
      %
      % NOTE: this function does not cache any intermediate states, and
      %       therefore operates independently from batchFilter() /
      %       batchSmooth() / extractMAP()
      %
      % NOTE: for a histogram inference engine, all probability density
      %       functions have been approximated by probability MASS
      %       functions, through sampling of dirac deltas + numerical
      %       normalization. Thus, to compute the joint probability of any
      %       specific state value, that value must first be mapped to the
      %       corresponding bin, then we extract the appropriate
      %       probability value from the NORMALIZED probability MASS
      %       function, as obtained using histogram binning.
      %
      %       Note that if we want to use logjointprob for evaluating
      %       the RELATIVE likelihood, then these normalization constants
      %       are not needed. This is the case when working within an EM
      %       framework.
      
      if isempty(obj.cache),
        obj.buildCache();
      end
      
      num_time_steps = size(data_all, 1);
      
      % Map states_all to bin centers
      state_diffs = abs(repmat(states_all, 1, obj.cache.num_bins) - ...
        repmat(obj.cache.x_vec, num_time_steps+1, 1));
      [~, state_bin_idx] = min(state_diffs, [], 2);
      state_binctr_all = obj.cache.x_vec(state_bin_idx)';
      
      % start with assumed uniform prior
      % NOTE: the PDF of each continuous x value is equivalent to their bin
      %       center PMF value, and these PMFs must sum up to num_bins in
      %       order for their corresponding histogram PDFs to integrate to 1
      logprobs = log(1);
      
      query.x_past = state_binctr_all(1);
      query.x_curr = obj.cache.x_vec';
      query.num_bins = obj.cache.num_bins;
      
      data_past = struct();
      for time = 1:num_time_steps,
        data_curr = data_all(time);

        query.eye_num_bins = obj.cache.eye_num_bins(:, state_bin_idx(time));

        logprobs_propagate = log(obj.model.propagate(query, data_curr, data_past));
        logprobs_observe = log(obj.model.observe(query, data_curr, data_past));
        logprobs_propobs = logprobs_propagate + logprobs_observe;
        lognorm_propobs = log(sum(exp(logprobs_propobs)));
        if lognorm_propobs == -inf,
          error('HistoEngine:pmfZero', ...
            'Evaluation of logJointProb failed: sum p(...) == 0!');
        else
          logprobs_propobs = logprobs_propobs - lognorm_propobs;
        end
        logprobs_propobs = logprobs_propobs(state_bin_idx(time+1));
        
        logprobs = logprobs + logprobs_propobs;
        if any(isnan(exp(logprobs_propobs))),
          error('HistoEngine:pmfZero', ...
            'Evaluation of logJointProb failed: p(...) has NaN term!');
        end
        
        query.x_past = state_binctr_all(time+1);
      end
    end % function logJointProb(...)
  end % methods
end % classdef HistoEngine

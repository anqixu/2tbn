classdef HistoEngineEM < HistoEngine
  properties (SetAccess='protected')
    data_all
    
    em
    % CONTENTS:
    % em.hard_em_type
    % em.iter % index
    % em.params_list
    % em.E_states_list % either list MAP trajectory, or matrix of smoothed
    % em.E_LJP_list % log joint prob
    % em.M_LJP_list
    % em.converged
  end
  
  
  
  properties
  end
  
  
  
  methods
    function obj=HistoEngineEM(model, init_params, num_bins, ...
        prior_x_pmf, data_all, hard_em_type)
      if nargin < 6,
        hard_em_type = 'map';
      end

      obj@HistoEngine(model, init_params, num_bins, prior_x_pmf);
      obj.data_all = data_all;
      obj.reset(init_params, hard_em_type);
    end
    
    
    function reset(obj, init_params, hard_em_type)
      if nargin < 2,
        init_params = obj.em.params_list{1};
      end
      if nargin < 3,
        hard_em_type = obj.em.hard_em_type;
      end
      
      obj.em.hard_em_type = hard_em_type;
      obj.em.iter = 0;
      obj.em.params_list = cell(1);
      obj.em.params_list{1} = init_params;
      obj.em.E_states_list = cell(0);
      obj.em.E_LJP_list = [];
      obj.em.M_LJP_list = [];
      obj.em.converged = false;
    end
    
    
    function err = runEM(obj, num_iters, iter_verbose, param_eps_gain, em_verbose)
      if nargin < 4,
        param_eps_gain = 1; % Use default param convergence epsilons
      end
      if nargin < 5,
        em_verbose = false;
      end
      
      err = 0;
      state_vec_matrix = repmat(obj.cache.x_vec, size(obj.data_all, 1)+1, 1);
      for i = 1:num_iters,
        if obj.em.converged,
          break;
        end
        
        if em_verbose,
          fprintf('> EM loop %3d / %3d, cur_iter = %3d\n', i, num_iters, obj.em.iter);
        end
        obj.em.iter = obj.em.iter + 1;
        
        % E step: infer latent states for current model
        try
          if strcmp(obj.em.hard_em_type, 'filter'),
            filtered_pmfs = obj.batchFilter(obj.data_all, iter_verbose);
            obj.em.E_states_list{obj.em.iter} = mean(filtered_pmfs.*state_vec_matrix, 2);
          elseif strcmp(obj.em.hard_em_type, 'smooth'),
            [smoothed_pmfs, ~] = obj.batchSmooth(obj.data_all, iter_verbose, false);
            obj.em.E_states_list{obj.em.iter} = mean(smoothed_pmfs.*state_vec_matrix, 2);
          else
            obj.batchFilter(obj.data_all, iter_verbose);
            obj.em.E_states_list{obj.em.iter} = obj.extractMAP();
          end
        catch err,
          if strcmp(err.identifier, 'HistoEngine:pmfZero'),
            warning('HistoEngineEM:pmfZero', ...
              'EM terminated prematurily on iter %d due to error: %s', ...
              obj.em.iter, err.message);
            break;
          else
            rethrow(err);
          end
        end
        obj.em.E_LJP_list(obj.em.iter) = ...
          obj.logJointProb(obj.data_all, obj.em.E_states_list{obj.em.iter});
        
        % M step: fit model parameters
        obj.em.params_list{obj.em.iter+1} = ...
          obj.model.optimizeParams(obj.data_all, ...
          obj.em.E_states_list{obj.em.iter}, ...
          obj.em.params_list{obj.em.iter});
        
        if obj.model.compareParams(obj.em.params_list{obj.em.iter}, ...
            obj.em.params_list{obj.em.iter+1}, param_eps_gain),
          obj.em.converged = true;
        end
        obj.updateParams(obj.em.params_list{obj.em.iter+1});
        obj.em.M_LJP_list(obj.em.iter) = ...
          obj.logJointProb(obj.data_all, obj.em.E_states_list{obj.em.iter});
      end
      
      if iter_verbose,
        fprintf('-> Total EM iters: %d (converged: %d)\n', ...
          obj.em.iter, obj.em.converged);
      end
    end % function runEM(...)
  end % methods
end % classdef HistoEngineEM

%% INPUT AND MODEL SETUP
% Load Interest Rate Data
%country_rates = readtable('it_interest_rate.csv'); country = 'Italy';
%country_rates = readtable('uk_interest_rate.csv'); country = 'United Kingdom';
country_rates = readtable('us_interest_rate.csv'); country = 'United States of America';

data = table2array(country_rates(:,2));
time = table2array(country_rates(:,1));
dt = 1; % Monthly Data

% Initialize Parameters
alpha = 0.1; 
beta = 0.1;  
sigma = 0.1; 
params_init = [alpha, beta, sigma];
g_constraint_vasicek = 0;
g_constraint_cir = 0.5;
optim_options = optimoptions('fminunc', 'Algorithm', 'quasi-newton');
rng(1)


%% VASICEK MODEL CALIBRATION
% Initial Estimation with Equal Weights
[param_est1_vasicek, Q_val1_vasicek] = fminunc(@(params) gmm_objective(params, data, g_constraint_vasicek, dt), params_init, optim_options);

% Compute HAC Weight Matrix and Re-estimate
W_hac_vasicek = compute_HAC(data, param_est1_vasicek, g_constraint_vasicek, dt);
[param_est2_vasicek, Q_val2_vasicek] = fminunc(@(params) gmm_objective(params, data, g_constraint_vasicek, dt, W_hac_vasicek), param_est1_vasicek, optim_options);

% Simulation of the Model
simulated_paths_vasicek = run_simulations(data, param_est2_vasicek, g_constraint_vasicek, dt);


%% CIR MODEL CALIBRATION
% Initial Estimation with Equal Weights
[param_est1_cir, Q_val1_cir] = fminunc(@(params) gmm_objective(params, data, g_constraint_cir, dt), params_init, optim_options);

% Compute HAC Weight Matrix and Re-estimate
W_hac_cir = compute_HAC(data, param_est1_cir, g_constraint_cir, dt);
[param_est2_cir, Q_val2_cir] = fminunc(@(params) gmm_objective(params, data, g_constraint_cir, dt, W_hac_cir), param_est1_cir, optim_options);

% Simulation of the Model
simulated_paths_cir = run_simulations(data, param_est2_cir, g_constraint_cir, dt);


%% PLOTS AND TABLES
% Display Calibrated Parameters
params_table = table({'Vasicek'; 'CIR'}, [param_est2_vasicek(1); param_est2_cir(1)], [param_est2_vasicek(2); param_est2_cir(2)], [param_est2_vasicek(3); param_est2_cir(3)], ...
    'VariableNames', {'Model', 'Alpha', 'Beta', 'Sigma'});
disp('----------------------------')
disp(sprintf('Estimated Parameters for %s:', country));
disp(params_table)

% Plot the Models
figure('Name', country)
subplot(1,2,1)
plot_model(country, time, data, simulated_paths_vasicek, g_constraint_vasicek)
subplot(1,2,2)
plot_model(country, time, data, simulated_paths_cir, g_constraint_cir)


%% VOLATILITY
[vol_actual, vol_vasicek, vol_cir] = calculate_volatility(data, simulated_paths_vasicek, simulated_paths_cir);

vol_table = table({'Historical Date'; 'Vasicek'; 'CIR'}, [vol_actual; vol_vasicek; vol_cir], ...
    'VariableNames', {'Model', 'Volatilty'});
disp('----------------------------')
disp(sprintf('Volatility Comparison for %s:', country));
disp(vol_table)


%% CHI2 TEST
critical_chi2_value = 3.841; % For 5 percent significance and 1 DoF
[chi2_stat_vasicek, p_value_vasicek, null_hypo_vasicek] = chi2_test(data, Q_val2_vasicek, critical_chi2_value);
[chi2_stat_cir, p_value_cir, null_hypo_cir] = chi2_test(data, Q_val2_cir, critical_chi2_value);

chi2_table = table({'Vasicek'; 'CIR'}, [chi2_stat_vasicek; chi2_stat_cir], [p_value_vasicek; p_value_cir], {null_hypo_vasicek; null_hypo_cir} ,...
    'VariableNames', {'Model', 'Test Statistic', 'p-Value', 'Null Hyphothesis'});
disp('----------------------------')
disp(sprintf('Chi^2 Test for %s:', country));
disp(chi2_table);


%% AUXILIARY FUNCTIONS
% GMM Objective Function
function Q = gmm_objective(params, data, g_constraint, dt, W)
    alpha = params(1);
    beta = params(2);
    sigma = params(3);

    % Residuals and Moments
    residuals = diff(data) - (alpha + beta * data(1:end-1)) * dt;
    moment1 = residuals; % First Moment
    moment2 = residuals.^2 - sigma^2 * (data(1:end-1).^g_constraint) * dt; % Second Moment
    moments = [moment1'; moment2'];

    if nargin < 5
        % Use Identity Matrix in First Step
        W = eye(size(moments, 1));
    end
    Q = mean(moments, 2)' * W * mean(moments, 2); % Objective Function
end


% Compute HAC Matrix
function W_opt = compute_HAC(data, param_est, g_constraint, dt)
    alpha = param_est(1);
    beta = param_est(2);
    sigma = param_est(3);

    % Residuals and Moments
    residuals = diff(data) - (alpha + beta * data(1:end-1)) * dt;
    moment1 = residuals; % First Moment
    moment2 = residuals.^2 - sigma^2 * (data(1:end-1).^g_constraint) * dt; % Second Moment
    moments = [moment1'; moment2']; 
    T = size(moments, 2); 

    % HAC Estimation
    lags = 12; 
    W = (moments * moments') / T; % Contemporaneous Covariance
    for lag = 1:lags
        gamma = (moments(:, 1:end-lag) * moments(:, lag+1:end)') / T; % Lagged Covariance
        W = W + (1 - lag / (lags + 1)) * (gamma + gamma'); % Bartlett Kernel
    end
    W_opt = inv(W); % Optimal Weights Matrix
end


% Simulation of the Models
function simulated_paths = run_simulations(data, param_est, g_constraint, dt)
    simulations = 1000; 
    forecast_horizon = length(data); 
    simulated_paths = zeros(forecast_horizon, simulations);
    
    for sim = 1:simulations
        sim_path = zeros(forecast_horizon, 1);
        sim_path(1) = data(1); 
        for t = 2:forecast_horizon
            dZ = sqrt(dt) * randn; % Brownian Motion Increment
            if g_constraint == 0
                % Vasicek Model
                sim_path(t) = sim_path(t-1) + ...
                    (param_est(1) + param_est(2) * sim_path(t-1)) * dt + ...
                    param_est(3) * dZ;
            else
                % CIR Model
                sim_path(t) = sim_path(t-1) + ...
                    (param_est(1) + param_est(2) * sim_path(t-1)) * dt + ...
                    param_est(3) * sqrt(max(sim_path(t-1), 0)) * dZ; % Used an absorbing boundary at r=0
            end
        end
        simulated_paths(:, sim) = sim_path;
    end
end


% Plot Historical and Simulated Rates
function plot_model(country, time, data, simulated_paths, g_constraint)
    mean_simulation = mean(simulated_paths, 2);
    conf_lower = quantile(simulated_paths, 0.16, 2);
    conf_upper = quantile(simulated_paths, 0.84, 2);


    plot(time, data, 'k-', 'LineWidth', 1.5);
    hold on;
    plot(time, mean_simulation, 'b--', 'LineWidth', 1.5); 
    plot(time, conf_lower, 'r--', 'LineWidth', 1); 
    plot(time, conf_upper, 'r--', 'LineWidth', 1); 
    legend('Historical Rates', 'Mean Simulated Rates', '68% Confidence Bounds');
    xlabel('Date');
    ylabel('Interest Rate (%)');
    grid on;
    if g_constraint == 0
        title('Vasicek Model');
    else
        title('CIR Model');
    end
    hold off;
    sgtitle(sprintf('Simulation Results for %s', country))
end


% Calculate Volatility
function [vol_actual, vol_vasicek, vol_cir] = calculate_volatility(data, simulated_paths_vasicek, simulated_paths_cir)
    vol_actual = std(diff(data));
    vol_vasicek = std(diff(mean(simulated_paths_vasicek, 2)));
    vol_cir = std(diff(mean(simulated_paths_cir, 2)));
end


% Chi-Square Test
function [test_statistic, p_value, null_hypo] = chi2_test(data, Q_value, critical_value)
    test_statistic = Q_value * length(data);
    p_value = 1- chi2cdf(test_statistic, 1);
    if test_statistic > critical_value
        null_hypo = 'Rejected';
    else
        null_hypo = 'Not Rejected';
    end
end

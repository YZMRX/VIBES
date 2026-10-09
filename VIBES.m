function [gamma, x, w, c, v, model] = VIBES(y, f, opts)
%AWSM_HVB_CHAMP_GPU Single-window hierarchical VB Champagne prototype.
%
%   [GAMMA, X, W, C, V, MODEL] = AWSM_HVB_CHAMP_GPU(Y, F, OPTS)
%   jointly estimates sparse sources and structured sensor noise under
%
%       Y = F * X + A * Z + E.
%
%   Source rows use voxel-wise ARD precisions.  The structured-noise
%   factors A and Z share component-wise ARD precisions, so unnecessary
%   candidate components are suppressed without fixing the final rank.
%   E is isotropic Gaussian residual noise with a learned precision.
%
%   This is the first, single-window implementation.  Temporal windowing
%   and AR state propagation should be implemented by a wrapper after this
%   core update has been validated.
%
%   Required inputs
%     y                 [n_channel x n_time] sensor data
%     f                 [n_channel x (nd*n_voxel)] lead field
%
%   The test configuration uses the fixed CPU path with a two-pass
%   augmented-ARD initialization, Cholesky residual factors, diagonal
%   residual noise precision, and posterior structured-noise uncertainty.
%
%   MODEL fields include noise_est, residual, and iterations.
%
%   Notes
%   -----
%   1. No explicit source/noise orthogonality constraint is imposed.
eps1       = 1e-10;
jitter     = 1e-9;
nd         = opts.nd;
max_iter   = opts.max_iter;
tol        = 1e-8;
init_champ_iter = 20;
iteration_plot_on = opts.iteration_plot_on;

[nk, nvd] = size(f);
nt = size(y, 2);

nv = nvd / nd;

% Main VB uses max_iter as an upper bound and stops when the ELBO stabilizes.
% The two-pass initializer uses a fixed internal iteration count.
init_champ_tol = max(sqrt(tol), 1e-4);

% Cholesky initialization retains the full sensor covariance; component ARD
% selects the effective rank during the VB iterations.
nm = nk;
y_work = double(y);
f_work = double(f);
% Per-voxel-group leadfield column normalization.
% Without this, shallow high-gain sources dominate and deep sources are
% suppressed by ARD.  Working on F_tilde = F_v / s_v makes the model
% scale-invariant; outputs are restored to physical units at the end.
fv_norm = zeros(nv, 1, 'like', y_work);
for iv = 1:nv
    cols = (iv - 1) * nd + (1:nd);
    fv_norm(iv) = norm(f_work(:, cols), 'fro');
end
fv_norm = max(fv_norm, eps1);
fv_norm_vec = reshape(ones(nd, 1, 'like', fv_norm) * fv_norm', [], 1);
f_work = f_work ./ fv_norm_vec';

eye_c = eye(nk, 'like', y_work);
eye_m = eye(nm, 'like', y_work);
log2pi = log(2 * pi);

% ---------------------------------------------------------------------
% Two-pass warm start: augmented ARD followed by residual covariance ARD.
% ---------------------------------------------------------------------
cyy0 = y_work * y_work' / nt;
sensor_power0 = max(real(trace(cyy0)) / nk, eps1);
lf_n = reshape(repmat(eye_c, nd, 1), nk, nk * nd);
    f_init = [f_work, lf_n];
    nvd_init = size(f_init, 2);
    nv_init = nvd_init / nd;

    f2_init = sum(abs(f_init).^2, 1);
    invf2_init = zeros(1, nvd_init, 'like', y_work);
    valid_f_init = f2_init > eps1;
    invf2_init(valid_f_init) = 1 ./ f2_init(valid_f_init);
    w0_init = f_init' .* invf2_init';
    x0_init = w0_init * y_work;
    inu0 = max(mean(abs(x0_init).^2, 'all'), eps1);
    vvec_init = inu0 * ones(nvd_init, 1, 'like', y_work);

    init_stall_count = 0;
    for init_iter = 1:init_champ_iter
        vvec_previous = vvec_init;
        c_init = (f_init .* vvec_init') * f_init';
        [rinv_y0, ~, L_c] = local_spd_solve(c_init, y_work, jitter);
        rinv_f0 = L_c' \ (L_c \ f_init);
        x0_init = vvec_init .* (f_init' * rinv_y0);
        x20_init = mean(abs(x0_init).^2, 2);
        zdiag0 = max(real(sum(conj(f_init) .* rinv_f0, 1)'), eps1);

        x2_group0 = sum(reshape(x20_init, nd, nv_init), 1);
        z_group0 = sum(reshape(zdiag0, nd, nv_init), 1);
        v_group0 = sqrt(max(x2_group0 ./ max(z_group0, eps1), 0));
        vvec_init = reshape(ones(nd, 1, 'like', y_work) * v_group0, nvd_init, 1);

        init_change = norm(vvec_init - vvec_previous) / ...
            max(norm(vvec_previous), eps1);
        if init_change < init_champ_tol
            init_stall_count = init_stall_count + 1;
        else
            init_stall_count = 0;
        end
        if init_stall_count >= 3
            break
        end
    end

    x0 = x0_init(1:nvd, :);

    residual_1st = y_work - f_work * x0;
        C_R = residual_1st * residual_1st' / nt;
        C_R = 0.5 * (C_R + C_R');
        [U_R, D_R] = eig(C_R);
        d_R = sort(max(real(diag(D_R)), 0), 'ascend');
        tail_count = max(floor(nk / 2), 1);
        sigma2_0 = median(d_R(1:tail_count));
        s_struct = max(d_R - sigma2_0, eps1);
        [~, idx_struct] = sort(s_struct, 'descend');
        nB_raw = min(nm, nnz(s_struct > 1e-3 * max(s_struct)));
        if nB_raw > 0
            nB = nd * ceil(nB_raw / nd);
        else
            nB = 0;
        end
        B = U_R(:, idx_struct(1:nB_raw));
        if nB > nB_raw
            B = [B, zeros(nk, nB - nB_raw, 'like', y_work)];
        end

        f_init2 = [f_work, B];
        nvd_init2 = size(f_init2, 2);
        f2_in2 = sum(abs(f_init2).^2, 1);
        invf2_in2 = zeros(1, nvd_init2, 'like', y_work);
        valid_f2 = f2_in2 > eps1;
        invf2_in2(valid_f2) = 1 ./ f2_in2(valid_f2);
        w0_in2 = f_init2' .* invf2_in2';
        x0_in2 = w0_in2 * y_work;
        inu_2 = max(mean(abs(x0_in2).^2, 'all'), eps1);
        vvec_in2 = inu_2 * ones(nvd_init2, 1, 'like', y_work);

        init2_stall_count = 0;
        for init_iter = 1:init_champ_iter
            vvec_previous = vvec_in2;
            c_in2 = (f_init2 .* vvec_in2') * f_init2';
            [rinv_y2, ~, L_c2] = local_spd_solve(c_in2, y_work, jitter);
            rinv_f2 = L_c2' \ (L_c2 \ f_init2);
            x0_in2 = vvec_in2 .* (f_init2' * rinv_y2);
            x20_in2 = mean(abs(x0_in2).^2, 2);
            zdiag_in2 = max(real(sum(conj(f_init2) .* rinv_f2, 1)'), eps1);
            x2g_in2 = sum(reshape(x20_in2, nd, nvd_init2/nd), 1);
            zg_in2 = sum(reshape(zdiag_in2, nd, nvd_init2/nd), 1);
            vg_in2 = sqrt(max(x2g_in2 ./ max(zg_in2, eps1), 0));
            vvec_in2 = reshape(ones(nd, 1, 'like', y_work) * vg_in2, nvd_init2, 1);

            init_change = norm(vvec_in2 - vvec_previous) / ...
                max(norm(vvec_previous), eps1);
        if init_change < init_champ_tol
                init2_stall_count = init2_stall_count + 1;
            else
                init2_stall_count = 0;
            end
            if init2_stall_count >= 3
                break
            end
        end

        x0 = x0_in2(1:nvd, :);
        v_group0 = mean(reshape(vvec_in2(1:nvd), nd, nv), 1)';
        noise_factor_two_pass = B * x0_in2(nvd+1:end, :);
alpha = 1 ./ max(v_group0, eps1);
x_work = x0;
residual0 = y_work - f_work * x_work;

% Estimate the initial white-noise precision from the lower half of the
% residual covariance spectrum; dominant structured modes stay in A*Z.
residual_cov0 = residual0 * residual0' / nt;
residual_eigs0 = sort(max(real(eig(0.5 * ...
    (residual_cov0 + residual_cov0'))), 0), 'ascend');
tail_count0 = max(floor(nk / 2), 1);
white_var0 = median(residual_eigs0(1:tail_count0));
residual_var0 = mean(abs(residual0).^2, 2);
residual_var0 = 0.5 * residual_var0 + 0.5 * white_var0;
tau_vec = 1 ./ max(residual_var0, sensor_power0 * eps1);

% Initialize A*Z from the source residual without discarding sensor modes.
noise_factor_init = noise_factor_two_pass;
[a_work, z_work] = local_initialize_noise_factors( ...
    noise_factor_init, eps1, jitter);

% Start the covariance corrections at zero; using identity matrices here
% would add nk/nt pseudo-observations and collapse the residual factor initialization.
sigma_a_stack = zeros(nm, nm, nk, 'like', y_work);
sigma_z = zeros(nm, nm, 'like', y_work);
a_energy = sum(abs(a_work).^2, 1)';
z_energy = sum(abs(z_work).^2, 2) + nt * real(diag(sigma_z));
lambda = (1e-6 + (nk + nt) / 2) ./ ...
    max(1e-6 + 0.5 * (a_energy + z_energy), eps1);

elbo = nan(max_iter, 1);
elbo_stall_count = 0;
if iteration_plot_on
    iteration_plot = local_init_iteration_plot(max_iter, nv);
else
    iteration_plot = struct();
end

for iter = 1:max_iter
    % ================================================================
    % q(X): source posterior conditioned on the current VB noise factors.
    % ================================================================
    noise_mean = a_work * z_work;
    y_clean = y_work - noise_mean;
    alpha_vec = reshape(ones(nd, 1, 'like', alpha) * alpha', [], 1);
    dsrc = 1 ./ max(alpha_vec, eps1);

    b_source = (f_work .* dsrc') * f_work';
    residual_var_vec = 1 ./ max(tau_vec, eps1);
    [~, structured_noise_uncertainty_cov] = ...
        local_noise_covariance_moments(a_work, z_work, sigma_z, ...
        sigma_a_stack, nt);
    noise_cov_for_source = structured_noise_uncertainty_cov + ...
        diag(residual_var_vec);
    [~, logdet_noise_cov, ~] = local_spd_solve( ...
        noise_cov_for_source, eye_c, jitter);
    r_source = b_source + noise_cov_for_source;
    [rinv_clean, logdet_r, L_r] = local_spd_solve(r_source, y_clean, jitter);
    rinv_f = L_r' \ (L_r \ f_work);

    x_work = dsrc .* (f_work' * rinv_clean);
    sigma_s_diag = real(dsrc - dsrc.^2 .* ...
        sum(conj(f_work) .* rinv_f, 1)');
    sigma_s_diag = max(sigma_s_diag, eps1);

    row_second_moment = sum(abs(x_work).^2, 2) + nt * sigma_s_diag;
    source_second_moment = sum(reshape(row_second_moment, nd, nv), 1)';
    alpha_shape = 1e-6 + nd * nt / 2;
    alpha_rate = 1e-6 + 0.5 * source_second_moment;
    alpha = alpha_shape ./ max(alpha_rate, eps1);
    % ================================================================
    % q(Z): structured-noise temporal coefficients.
    % ================================================================
    source_residual = y_work - f_work * x_work;
    weighted_a = bsxfun(@times, a_work, tau_vec);
    expected_a_t_a = a_work' * weighted_a;
    sum_tau_sigma_a = sum(bsxfun(@times, sigma_a_stack, reshape(tau_vec, 1, 1, nk)), 3);
    precision_z = expected_a_t_a + sum_tau_sigma_a + diag(lambda);
    [sigma_z, ~] = ...
        local_spd_solve(precision_z, eye_m, jitter);
    z_work = sigma_z * (a_work' * bsxfun(@times, source_residual, tau_vec));
    expected_zzt = z_work * z_work' + nt * sigma_z;
    % ================================================================
    % q(A): structured-noise spatial basis.
    % ================================================================
    sigma_a_stack = zeros(nm, nm, nk, 'like', y_work);
    a_work_new = zeros(nk, nm, 'like', y_work);
    a_energy = zeros(nm, 1, 'like', y_work);
    logdet_sigma_a_total = 0;
    for ic = 1:nk
        P = tau_vec(ic) * expected_zzt + diag(lambda);
        P = 0.5 * (P + P');
        [L, p] = chol(P, 'lower');
        if p > 0
            P = P + (jitter * max(real(trace(P)), 1) / nm) * eye_m;
            [L, p] = chol(P, 'lower');
            if p > 0
                error('awsm_hvb_champ_gpu:NotPositiveDefinite', ...
                    'Per-channel precision remained indefinite.');
            end
        end
        sigma_a_ic = L' \ (L \ eye_m);
        sigma_a_stack(:, :, ic) = sigma_a_ic;
        a_work_new(ic, :) = tau_vec(ic) * ...
            (source_residual(ic, :) * z_work') * sigma_a_ic;
        a_energy = a_energy + (abs(a_work_new(ic, :)).^2)' + ...
            real(diag(sigma_a_ic));
        logdet_sigma_a_total = logdet_sigma_a_total ...
            - 2 * sum(log(max(real(diag(L)), eps1)));
    end
    a_work = a_work_new;
    % ================================================================
    % Symmetric Gauss--Seidel q(Z) back sweep.
    %
    % The forward q(Z)->q(A) pass leaves Z conditioned on the previous A.
    % Recomputing its conjugate posterior after q(A) makes the structured
    % noise A*Z used by the next source step consistent with the latest
    % spatial factor.  This is a second exact coordinate update of the
    % same ELBO, with no added prior, tuning parameter, or data change.
    weighted_a = bsxfun(@times, a_work, tau_vec);
    expected_a_t_a = a_work' * weighted_a;
    sum_tau_sigma_a = sum(bsxfun(@times, sigma_a_stack, ...
        reshape(tau_vec, 1, 1, nk)), 3);
    precision_z = expected_a_t_a + sum_tau_sigma_a + diag(lambda);
    [sigma_z, logdet_precision_z] = ...
        local_spd_solve(precision_z, eye_m, jitter);
    z_work = sigma_z * (a_work' * bsxfun(@times, source_residual, tau_vec));
    expected_zzt = z_work * z_work' + nt * sigma_z;
    z_energy = real(diag(expected_zzt));

    % Shared component precision removes the A/Z scale ambiguity and
    % suppresses redundant candidate noise components.
    lambda_shape = 1e-6 + (nk + nt) / 2;
    lambda_rate = 1e-6 + 0.5 * (a_energy + z_energy);
    lambda = lambda_shape ./ max(lambda_rate, eps1);
    % ================================================================
    % q(tau): residual white-noise precision with uncertainty corrections.
    % ================================================================
    noise_mean = a_work * z_work;
    mean_residual = y_work - f_work * x_work - noise_mean;

    LrB = L_r \ b_source;
    source_uncertainty_by_channel = nt * ...
        max(real(diag(b_source)) - sum(abs(LrB).^2, 1)', 0);

    a_expected_zzt_a = real(sum((a_work * expected_zzt) .* conj(a_work), 2));
    trace_sigma_zzt = real(sum(sum(bsxfun(@times, sigma_a_stack, expected_zzt.'), 1), 2));
    trace_sigma_zzt = reshape(trace_sigma_zzt, nk, 1);
    expected_noise_by_ch = a_expected_zzt_a + trace_sigma_zzt;
    mean_noise_by_ch = sum(abs(noise_mean).^2, 2);
    noise_uncertainty_by_channel = max(expected_noise_by_ch - mean_noise_by_ch, 0);

    expected_residual = sum(abs(mean_residual).^2, 2) + ...
        source_uncertainty_by_channel + noise_uncertainty_by_channel;
    tau_shape = 1e-6 + nt / 2;
    tau_rate = 1e-6 + 0.5 * expected_residual;
    tau_vec = tau_shape ./ max(tau_rate, eps1);
    tau = mean(tau_vec);

    % ================================================================
    % Uncertainty-aware variational objective for fixed-iteration monitoring.
    % ================================================================
    tau_for_elbo = tau_vec;
    logdet_sigma_s = sum(log(max(dsrc, eps1))) ...
        + logdet_noise_cov - logdet_r;
    logdet_sigma_a = logdet_sigma_a_total;
    logdet_sigma_z = -logdet_precision_z;

    er_cpu = expected_residual;
    ssm_cpu = source_second_moment;
    ae_cpu = a_energy;
    ze_cpu = z_energy;
    lds_cpu = logdet_sigma_s;
    lda_cpu = logdet_sigma_a;
    ldz_cpu = logdet_sigma_z;
    al_cpu = alpha;
    alr_cpu = alpha_rate;
    lm_cpu = lambda;
    lmr_cpu = lambda_rate;
    tu_cpu = tau_for_elbo;
    tur_cpu = tau_rate;

    elog_tau = psi(tau_shape) - log(tur_cpu);
    elog_alpha = psi(alpha_shape) - log(alr_cpu);
    elog_lambda = psi(lambda_shape) - log(lmr_cpu);

    likelihood = 0.5 * nt * (sum(elog_tau(:)) - nk * log2pi) ...
        - 0.5 * sum(tu_cpu(:) .* er_cpu(:));

    source_prior = 0.5 * nd * nt * sum(elog_alpha - log2pi) ...
        - 0.5 * sum(al_cpu .* ssm_cpu);
    source_entropy = 0.5 * nt * (nvd * (1 + log2pi) + lds_cpu);
    noise_prior = 0.5 * (nk + nt) * sum(elog_lambda - log2pi) ...
        - 0.5 * sum(lm_cpu .* (ae_cpu + ze_cpu));
    a_entropy = 0.5 * (nk * nm * (1 + log2pi) + lda_cpu);
    z_entropy = 0.5 * nt * (nm * (1 + log2pi) + ldz_cpu);

    s0 = 1e-6;  r0 = 1e-6;
    alpha_terms = sum(s0*log(r0) - gammaln(s0) + (s0-1).*elog_alpha - r0.*al_cpu ...
        + alpha_shape - log(alr_cpu) + gammaln(alpha_shape) + (1-alpha_shape).*psi(alpha_shape));
    lambda_terms = sum(s0*log(r0) - gammaln(s0) + (s0-1).*elog_lambda - r0.*lm_cpu ...
        + lambda_shape - log(lmr_cpu) + gammaln(lambda_shape) + (1-lambda_shape).*psi(lambda_shape));
    tau_terms = sum(s0*log(r0) - gammaln(s0) + (s0-1).*elog_tau - r0.*tu_cpu ...
        + tau_shape - log(tur_cpu) + gammaln(tau_shape) + (1-tau_shape).*psi(tau_shape));
    elbo(iter) = real(likelihood + source_prior + source_entropy + noise_prior ...
        + a_entropy + z_entropy + alpha_terms + lambda_terms + tau_terms);

    x_physical_iter = x_work ./ fv_norm_vec;
    voxel_energy_iter = reshape(abs(x_physical_iter).^2, nd, nv, nt);
    voxel_power_iter = sum(sum(voxel_energy_iter, 1), 3);
    voxel_power_iter = reshape(voxel_power_iter, nv, 1);

    if iteration_plot_on
        local_update_iteration_plot(iteration_plot, iter, elbo(iter), ...
            voxel_power_iter);
    end

    if iter > 1
        elbo_change = abs(elbo(iter) - elbo(iter - 1)) / ...
            max(abs(elbo(iter - 1)), 1);
        if elbo_change < tol
            elbo_stall_count = elbo_stall_count + 1;
        else
            elbo_stall_count = 0;
        end
        if elbo_stall_count >= 3
            break
        end
    end

end

n_iter = iter;

% Recompute all public outputs from the final hyperparameters.  This avoids
% the stale-W/X issue in the original GPU implementation.
alpha_vec = reshape(ones(nd, 1, 'like', alpha) * alpha', [], 1);
dsrc = 1 ./ max(alpha_vec, eps1);
noise_mean = a_work * z_work;
b_source = (f_work .* dsrc') * f_work';
[structured_noise_cov, structured_noise_uncertainty_cov] = ...
    local_noise_covariance_moments(a_work, z_work, sigma_z, ...
    sigma_a_stack, nt);
residual_noise_cov = diag(1 ./ max(tau_vec, eps1));
% The public source filter is the conditional variational mean associated
% with the final q(A)q(Z) coordinate solution.
noise_cov_for_source = structured_noise_uncertainty_cov + residual_noise_cov;
source_cov_work = b_source;
r_source = b_source + noise_cov_for_source;
[rinv_clean, ~, L_r_out] = local_spd_solve(r_source, y_work - noise_mean, jitter);
rinv_eye = L_r_out' \ (L_r_out \ eye_c);
w_work = dsrc .* (f_work' * rinv_eye);
x_work = dsrc .* (f_work' * rinv_clean);

v_group = 1 ./ max(alpha, eps1);

% Restore physical units: F = F_tilde .* s  =>  X = X_tilde ./ s
x_work = x_work ./ fv_norm_vec;
w_work = w_work ./ fv_norm_vec;
v_group = v_group ./ max(fv_norm.^2, eps1);

v = (nd * v_group)';
x = x_work;
w = w_work;
a_cpu = a_work;
z_cpu = z_work;
noise_est = a_cpu * z_cpu;

noise_cov = structured_noise_cov + residual_noise_cov;
source_cov = source_cov_work;
c = real(source_cov + noise_cov);
c = 0.5 * (c + c');

gamma = zeros(nd, nd, nv);
v_group_cpu = v_group;
for iv = 1:nv
    gamma(:, :, iv) = v_group_cpu(iv) * eye(nd);
end

source_sensor = f * x;

model = struct();
model.noise_est = noise_est;
model.residual = y - source_sensor - noise_est;
model.iterations = n_iter;

end

% =====================================================================
% Local utilities
% =====================================================================

function [a_init, z_init] = local_initialize_noise_factors( ...
    residual, eps1, jitter)
[nk, nt] = size(residual);

residual_cov = residual * residual' / nt;
residual_cov = 0.5 * (residual_cov + residual_cov');
cov_scale = max(real(trace(residual_cov)) / nk, eps1);
eye_c = eye(nk, 'like', residual);
ridge0 = jitter * cov_scale;
for attempt = 0:7
    ridge = (10^attempt) * ridge0;
    [a_init, p] = chol(residual_cov + ridge * eye_c, 'lower');
    if p == 0
        break
    end
end
if p ~= 0
    error('awsm_hvb_champ_gpu:CholeskyInitFailed', ...
        'Residual covariance could not be regularized for Cholesky initialization.');
end
z_init = a_init \ residual;
[a_init, z_init] = local_balance_noise_factors(a_init, z_init, eps1);
end


function [a_factor, z_factor] = local_balance_noise_factors(a_factor, z_factor, eps1)
% Balance A/Z column-row energies while preserving their product exactly.
a_norm = sqrt(sum(abs(a_factor).^2, 1))';
z_norm = sqrt(sum(abs(z_factor).^2, 2));
scale = sqrt(max(z_norm, eps1) ./ max(a_norm, eps1));
a_factor = a_factor .* scale';
z_factor = z_factor ./ scale;
end


function [structured_cov, uncertainty_cov] = local_noise_covariance_moments( ...
    a_mean, z_mean, sigma_z, sigma_a_stack, nt)
% Expected sensor covariance of A*Z, split into mean and posterior variance.
mean_noise = a_mean * z_mean;
mean_cov = mean_noise * mean_noise' / nt;
z_second_moment = z_mean * z_mean' / nt + sigma_z;

% q(Z) contributes full sensor covariance; independent q(A_i) rows add
% only sensor-wise uncertainty, which belongs on the diagonal.
uncertainty_cov = a_mean * sigma_z * a_mean';
nk = size(a_mean, 1);
a_uncertainty_var = real(sum(sum(bsxfun(@times, sigma_a_stack, ...
    z_second_moment.'), 1), 2));
a_uncertainty_var = reshape(a_uncertainty_var, nk, 1);
uncertainty_cov = uncertainty_cov + diag(max(a_uncertainty_var, 0));
uncertainty_cov = 0.5 * (uncertainty_cov + uncertainty_cov');

structured_cov = mean_cov + uncertainty_cov;
structured_cov = 0.5 * (structured_cov + structured_cov');
end


function [solution, logdet_a, L] = local_spd_solve(a, b, jitter)
a = 0.5 * (a + a');
n = size(a, 1);
scale = max(real(trace(a)) / max(n, 1), 1);
eye_n = eye(n, 'like', a);

for attempt = 0:7
    if attempt == 0
        a_try = a;
    else
        a_try = a + (10^(attempt - 1) * jitter * scale) * eye_n;
    end
    [lower_a, p] = chol(a_try, 'lower');
    if p == 0
        solution = lower_a' \ (lower_a \ b);
        logdet_a = 2 * sum(log(real(diag(lower_a))));
        if nargout > 2
            L = lower_a;
        end
        return
    end
end

error('awsm_hvb_champ_gpu:NotPositiveDefinite', ...
    'A posterior precision matrix remained indefinite after jittering.');
end




function plot_state = local_init_iteration_plot(max_iter, num_voxels)
fig = figure('Color', 'w', 'Name', 'VIBES iterations', ...
    'NumberTitle', 'off');
fig.Position(3:4) = [600 260];
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax_elbo = nexttile;
hold(ax_elbo, 'on');
grid(ax_elbo, 'on');
xlim(ax_elbo, [1 max_iter]);
xlabel(ax_elbo, 'Iteration');
ylabel(ax_elbo, 'ELBO');
title(ax_elbo, 'Variational objective');

ax_energy = nexttile;
hold(ax_energy, 'on');
grid(ax_energy, 'on');
xlim(ax_energy, [1 num_voxels]);
ylim(ax_energy, [0 1]);
xlabel(ax_energy, 'Voxel index');
ylabel(ax_energy, 'Normalized power');
title(ax_energy, 'VIBES');

plot_state = struct();
plot_state.fig = fig;
plot_state.elbo = animatedline(ax_elbo, 'Color', [0.10 0.25 0.75], ...
    'LineWidth', 1.4, 'DisplayName', 'ELBO');
plot_state.voxel_power = plot(ax_energy, 1:num_voxels, zeros(num_voxels, 1), ...
    'Color', [1 0 0], 'LineWidth', 1.2);
drawnow;
end


function local_update_iteration_plot(plot_state, iter, elbo_value, ...
    voxel_power)
if ~isgraphics(plot_state.fig)
    return;
end

addpoints(plot_state.elbo, iter, elbo_value);
voxel_power = real(double(voxel_power(:)));
voxel_power(~isfinite(voxel_power) | voxel_power < 0) = 0;
maximum_power = max(voxel_power);
if maximum_power > 0
    voxel_power = voxel_power / maximum_power;
end
set(plot_state.voxel_power, 'YData', voxel_power);
drawnow limitrate;
end

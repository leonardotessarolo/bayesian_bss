import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import itertools

import scipy.stats as st

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns

from .source import StudentTSource, StandardLogisticSource, LogisticSource, TriangularSource
from .prior import MultivariateTPrior, ExponentialPrior

class PosteriorUtilities:

    @staticmethod
    def get_log_posterior_fn(
        source_pdf_fn,
        prior_pdf_fn,
        use_jax=False
    ):
        """
            This method returns a method for calculating log-posterior inside MCMC, with the option of using a jax
            backend or not.
        """
        if not use_jax:
            def __log_posterior_fn(
                x,
                B,
                source_pdf_fn,
                prior_pdf_fn
            ):
                NOBS=x.shape[-1]
                # Cálculo de posteriori para registros
                priori = prior_pdf_fn(B)
                if priori > 0:
                    posteriori = NOBS*np.log(np.abs(np.linalg.det(B))) + np.log(priori)
                    y=B@x
                    iterator = np.ndindex(x.shape)
                    posteriori += np.sum(np.log(
                        np.array([
                            source_pdf_fn(y[i,j]) for i,j in iterator
                        ])
                    ))
                elif priori == 0:
                    posteriori = -np.inf
                else:
                    print(priori)
                    print(B)
                    print(prior_pdf_fn)
                    raise ValueError('Prior value must be non-negative.')
                return posteriori

            return lambda x, B: __log_posterior_fn(
                x=x,
                B=B,
                source_pdf_fn=source_pdf_fn,
                prior_pdf_fn=prior_pdf_fn
            )
        else:
            def __log_posterior_fn(
                x,
                B,
                source_pdf_fn,
                prior_pdf_fn
            ):
                x = jnp.asarray(x)
                B = jnp.asarray(B)

                NOBS=x.shape[-1]

                # Avaliar priori
                priori = prior_pdf_fn(B)
                
                # Avaliar determinante
                sign, logabsdet = jnp.linalg.slogdet(B)
                
                # y = B@x
                y=jnp.matmul(B,x)
                
                # Avaliar somatório
                log_source = jnp.sum(
                    jnp.log(source_pdf_fn(y)),
                    axis=(-2, -1)
                )
                log_lik = NOBS*logabsdet + log_source
                
                # Log-priori segura: o duplo `where` impede gradientes NaN vindos de
                # log(0), pois o JAX avalia os dois ramos antes de selecionar.
                safe_priori = jnp.where(priori > 0, priori, 1.0)
                log_prior = jnp.where(priori > 0, jnp.log(safe_priori), -jnp.inf)
                posteriori = jnp.where(priori > 0, log_lik + log_prior, -jnp.inf)

                return posteriori

            return lambda x, B: __log_posterior_fn(
                x=x,
                B=B,
                source_pdf_fn=source_pdf_fn,
                prior_pdf_fn=prior_pdf_fn
            )


class AnalysisUtilities:

    SYM = np.array([[0., 1.], [1., 0.]])     # S: must match contour_line.py
    SKEW = np.array([[0., -1.], [1., 0.]])   # K: must match contour_line.py

    # ------------------------------------------------------------------ #
    # Projection onto the affine plane  B* + span{S, K}
    # ------------------------------------------------------------------ #
    @staticmethod
    def project_uv(B, B_true, S=None, K=None):
        """
        Orthogonal (Frobenius) projection of B onto {B_true + u S + v K}.
        B may be (n, n) or (R, n, n).
        Returns u, v, B_proj, dist_to_plane (= ||B - B_proj||_F).
        """
        S = AnalysisUtilities.SYM if S is None else S
        K = AnalysisUtilities.SKEW if K is None else K
        B = np.asarray(B, dtype=float)
        single = B.ndim == 2
        if single:
            B = B[None]

        D = (B - B_true).reshape(B.shape[0], -1)              # (R, n^2)
        M = np.stack([S.ravel(), K.ravel()], axis=1)          # (n^2, 2)
        coef, *_ = np.linalg.lstsq(M, D.T, rcond=None)        # (2, R)  -- general Gram solve
        u, v = coef
        B_proj = B_true + u[:, None, None] * S + v[:, None, None] * K
        dist = np.linalg.norm(B - B_proj, axis=(1, 2))

        if single:
            return u[0], v[0], B_proj[0], dist[0]
        return u, v, B_proj, dist

    # ------------------------------------------------------------------ #
    # Main comparison
    # ------------------------------------------------------------------ #
    @staticmethod
    def compare_average_B(
        parser,
        only_converged_mmse: bool = False,
        decimals: int = 3,
        verbose: bool = True,
    ):
        """
        For each test case:
          - average B over realizations for Grid / MAP / MMSE
          - projection (u, v) of each estimate onto the plane B* + uS + vK
          - distance of each estimate to that plane (part the grid cannot see)

        Returns
        -------
        summary : DataFrame indexed by (test_case, method) with
                  b11..b22, u, v, u_std, v_std, dist_to_plane, mean_dist_to_plane,
                  ||B-B*||_F, delta_B, mean_delta_B, n
        per_realization : long DataFrame (test_case, method, realization, u, v,
                          dist_to_plane, delta_B), ready for scatter plots
        """
        if not hasattr(parser, "parsed_results"):
            raise AttributeError("Call parser.parse() before compare_average_B().")

        B_true = np.asarray(parser.execution_config["general"]["B"], dtype=float)
        if B_true.shape != (2, 2):
            raise ValueError("The (u, v) plane is defined for 2x2 B only.")
        norm_true = np.linalg.norm(B_true)

        methods = {"Grid": "posteriori_grid", "MAP": "map", "MMSE": "mmse"}

        rows, per_real_rows, blocks = [], [], {}
        for tc in parser.execution_config["sim"]["test_cases"].keys():
            res_tc = parser.parsed_results[tc]
            realizations = np.asarray(res_tc["realization"])
            blocks[tc] = {"B*": (B_true, None, (0.0, 0.0, 0.0))}

            for label, key in methods.items():
                mats = np.stack([np.asarray(B, dtype=float)
                                 for B in res_tc[key]["B_estimates"]])     # (R, 2, 2)

                mask = np.all(np.isfinite(mats), axis=(1, 2))
                if label == "MMSE" and only_converged_mmse:
                    mask &= np.asarray(res_tc["mmse"]["converged"], dtype=bool)
                mats, reals = mats[mask], realizations[mask]
                n = mats.shape[0]

                if n == 0:
                    B_avg = np.full_like(B_true, np.nan)
                    u_r = v_r = d_r = err_r = np.array([np.nan])
                    u_a = v_a = d_a = np.nan
                else:
                    B_avg = mats.mean(axis=0)
                    # per-realization projections
                    u_r, v_r, _, d_r = AnalysisUtilities.project_uv(mats, B_true)
                    err_r = np.linalg.norm(mats - B_true, axis=(1, 2)) / norm_true
                    # projection of the averaged matrix (= mean of u_r, v_r by linearity)
                    u_a, v_a, _, d_a = AnalysisUtilities.project_uv(B_avg, B_true)

                    for rz, uu, vv, dd, ee in zip(reals, u_r, v_r, d_r, err_r):
                        per_real_rows.append({
                            "test_case": tc, "method": label, "realization": rz,
                            "u": uu, "v": vv, "dist_to_plane": dd, "delta_B": ee,
                        })

                err = np.linalg.norm(B_avg - B_true)
                blocks[tc][label] = (B_avg, n, (u_a, v_a, d_a))

                rows.append({
                    "test_case": tc, "method": label,
                    "b11": B_avg[0, 0], "b12": B_avg[0, 1],
                    "b21": B_avg[1, 0], "b22": B_avg[1, 1],
                    "u": u_a, "v": v_a,
                    "u_std": np.nanstd(u_r), "v_std": np.nanstd(v_r),
                    "dist_to_plane": d_a,                 # of the averaged matrix
                    "mean_dist_to_plane": np.nanmean(d_r),
                    "||B-B*||_F": err,
                    "delta_B": err / norm_true,
                    "mean_delta_B": np.nanmean(err_r),
                    "n": n,
                })

        summary = pd.DataFrame(rows).set_index(["test_case", "method"]).round(decimals)
        per_realization = pd.DataFrame(per_real_rows)

        if verbose:
            AnalysisUtilities._print_side_by_side(blocks, decimals)

        return summary, per_realization

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_side_by_side(blocks: dict, decimals: int = 3) -> None:
        S, K = AnalysisUtilities.SYM, AnalysisUtilities.SKEW
        if not np.isclose(np.sum(S * K), 0.0):
            raise ValueError("S and K must be Frobenius-orthogonal for the per-axis formulas.")
        p, w = decimals, decimals + 5
        num = lambda x: f"{x:+.{p}f}"
        mat_row = lambda r: "[" + " ".join(f"{x:>{w}.{p}f}" for x in r) + " ]"

        def nz(M):
            return [(i, j) for i, j in np.ndindex(M.shape) if M[i, j] != 0]

        def symbolic(M):
            s = ""
            for i, j in nz(M):
                c = M[i, j]
                mag = "" if abs(c) == 1 else f"{abs(c):g}·"
                s += f"{'+' if c > 0 else '-'}{mag}Δ{i+1}{j+1}"
            return s.lstrip("+")

        def coord_line(name, M, dB):
            den = np.sum(M ** 2)
            terms = " ".join(num(M[i, j] * dB[i, j]) for i, j in nz(M))
            val = np.sum(M * dB) / den
            return f"{name}=({terms})/{den:g}={num(val)}", val

        # Legend, printed once
        print("Projection onto B* + uS + vK  (Frobenius inner product, S ⟂ K):")
        print(f"  ΔB = B̂ − B*")
        print(f"  u  = ⟨ΔB,S⟩/‖S‖² = ({symbolic(S)})/{np.sum(S**2):g}")
        print(f"  v  = ⟨ΔB,K⟩/‖K‖² = ({symbolic(K)})/{np.sum(K**2):g}")
        print(f"  d  = ‖ΔB − uS − vK‖_F   (part of B̂ outside the plane)")

        B_true = next(iter(blocks.values()))["B*"][0]
        for tc, mats in blocks.items():
            headers, cols = [], []
            for name, (B, n, _) in mats.items():
                headers.append(name if n is None else f"{name} (n={n})")
                lines = [mat_row(B[0]), mat_row(B[1])]
                if name == "B*" or not np.all(np.isfinite(B)):
                    lines += [""] * 5
                else:
                    dB = B - B_true
                    lines += ["ΔB=" + mat_row(dB[0]), "   " + mat_row(dB[1])]
                    u_line, u = coord_line("u", S, dB)
                    v_line, v = coord_line("v", K, dB)
                    R = dB - u * S - v * K
                    res = [R[i, j] for i, j in np.ndindex(R.shape) if abs(R[i, j]) > 1e-12]
                    d = np.linalg.norm(R)
                    d_line = ("d=0" if not res else
                            "d=√(" + "+".join(f"{abs(r):.{p}f}²" for r in res) + f")={d:.{p}f}")
                    lines += [u_line, v_line, d_line]
                cols.append(lines)
            col_w = max(max(len(s) for s in L) for L in cols)
            col_w = max(col_w, max(len(h) for h in headers)) + 3
            print(f"\n=== Test case {tc} ===")
            print("".join(h.ljust(col_w) for h in headers))
            for k in range(len(cols[0])):
                print("".join(L[k].ljust(col_w) for L in cols))

    @staticmethod
    def get_test_case_pairs(
        method
    ):
        test_pairs = []

        if method=='same_source':
            # Add perfect model cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['i','ii','iii','iv'], 2)
            ]

            # Add slightly misspecified model cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['v','vi','vii','viii'], 2)
            ]

            # Add largely misspecified model cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['ix','x','xi','xii'], 2)
            ]

        elif method=='same_prior':
            # Add uniform prior cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['i','v','ix'], 2)
            ]

            # Add low information correct prior cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['ii','vi','x'], 2)
            ]

            # Add high information correct prior cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['iii','vii','xi'], 2)
            ]

            # Add high information incorrect prior cases
            test_pairs = test_pairs + [
                comb for comb in itertools.combinations(['iv','viii','xii'], 2)
            ]
        else:
            raise ValueError("'method must be one of 'same_source' or 'same_prior'. ")
        
        return test_pairs
    

class LocalSensitivityScores:
    """
        Escores (derivadas do log da densidade) para sensibilidade local.
        score(samples, x, baseline_prior, baseline_source_model) -> array (K,)
        Derivada em relação ao PRÓPRIO hiperparâmetro (alpha, sigma_p, nu_p,
        mu_s, sigma_s, nu_s), no baseline. Sem derivadas em log(parâmetro)
        nem em multiplicadores relativos (p = sigma/sigma0, r = nu/nu0).
        Constantes omitidas (não afetam covariâncias):
        sigma_p : -d/sigma_p
        nu_p : 1/2[psi((nu+d)/2) - psi(nu/2) - d/nu] - d/(nu^2 sigma^2)
        sigma_s : -L*N/sigma_s
        nu_s : L*N{1/2[psi((nu+1)/2) - psi(nu/2) - 1/nu] - 1/(nu^2 sigma^2)}
    """
    @staticmethod
    def _prior_terms(
        samples,
        prior
    ):
        # scale escalar: matriz de forma Sigma = scale^2 I (não é desvio-padrão)
        center = np.asarray(prior.center, dtype=np.float64)
        scale, df = float(prior.scale), float(prior.df)

        # Obtém amostras centralizadas pela média da prior
        R = samples - center
        
        # Argumento da Student-t
        r = (R ** 2).sum(
            axis=(1, 2)
        ) / scale ** 2

        return R, r, scale, df, center.size
    

    @staticmethod
    def _source_terms(
        samples,
        x,
        source,
        chunk_size=500
    ):
        # Convert to float
        mu, sigma, nu = float(source.mu), float(source.sigma), float(source.df)
        
        # Total samples
        K = samples.shape[0]
        
        # Initialize dict for storing terms
        out = {
            k: np.empty(K) for k in (
                'mu',
                'sigma',
                'nu_partial'
            )
        }

        # Iterate over chunks
        for a in range(0, K, chunk_size):
            # Arguments for scores
            z = (
                np.einsum(
                    'kij,jn->kin',
                    samples[a:a + chunk_size], x
                ) - mu
            ) / sigma
            q = nu + z ** 2

            # d/d mu (completo)
            out['mu'][a:a + chunk_size] = (
                (nu + 1) * z / (sigma * q)
            ).sum(axis=(1, 2))

            # d/d sigma, sem a constante -L*N/sigma
            out['sigma'][a:a + chunk_size] = (
                (nu + 1) * z ** 2 / (sigma * q)
            ).sum(axis=(1, 2))

            # d/d nu com sigma fixo, sem os termos de digama
            out['nu_partial'][a:a + chunk_size] = (
                -0.5 * np.log1p(z ** 2 / nu) + (nu + 1) * z ** 2 / (2 * nu * q)
            ).sum(
                axis=(1, 2)
            )

        return out, sigma, nu
    

    @staticmethod
    def _unit_variance_dsigma_dnu(sigma, nu):
        # sigma(nu) = sqrt((nu-2)/nu) => dsigma/dnu = 1/(nu^2 sigma)
        if nu <= 2:
            raise ValueError('Trajetória de variância unitária exige df > 2.')
        return 1.0 / (nu ** 2 * sigma)
    

    # ---------------- Priori ----------------
    @staticmethod
    def prior_mode_score(
        direction
    ):
        # d/d alpha, center(alpha) = (1-alpha) B + alpha I => direction = I - B
        direction = np.asarray(direction, dtype=np.float64)

        def __prior_mode_score(
            samples,
            x,
            baseline_prior,
            baseline_source_model
        ):
            # Obtain prior terms
            R, r, s, nu_B, L2 = LocalSensitivityScores._prior_terms(
                samples, 
                baseline_prior
            )

            grad_center = ((nu_B + L2) / (nu_B + r))[:, None, None] * R / s ** 2

            return np.tensordot(
                grad_center,
                direction,
                axes=([1, 2], [0, 1])
            )
        
        return __prior_mode_score
    

    @staticmethod
    def prior_scale_score(
        samples,
        x,
        baseline_prior,
        baseline_source_model
    ):
        # d/d sigma_p, sem a constante -d/sigma_p
        R, r, s, nu, d = LocalSensitivityScores._prior_terms(
            samples,
            baseline_prior
        )
        
        return (nu + d) * r / (s * (nu + r))
    
    @staticmethod
    def prior_df_score(
        samples,
        x,
        baseline_prior,
        baseline_source_model
    ):
        """d/d nu_p ao longo de sigma_p(nu_p) = sqrt((nu_p-2)/nu_p)."""
        R, r, s, nu, d = LocalSensitivityScores._prior_terms(
            samples, 
            baseline_prior
        )
        # Pure df variation
        df_score = -0.5 * np.log1p(r / nu) + (nu + d) * r / (2 * nu * (nu + r))
        
        # Pure scale variation
        scale_score = (nu + d) * r / (s * (nu + r)) # d/d sigma_p

        return df_score + scale_score * LocalSensitivityScores._unit_variance_dsigma_dnu(s, nu)
    

    # ------------- Modelo das fontes -------------


    @staticmethod
    def source_mu_score(
        samples,
        x,
        baseline_prior,
        baseline_source_model
    ):
        # d/d mu_s
        return LocalSensitivityScores._source_terms(
            samples,
            x,
            baseline_source_model
        )[0]['mu']
    

    @staticmethod
    def source_sigma_score(
        samples,
        x,
        baseline_prior,
        baseline_source_model
    ):
        """d/d sigma_s (escala absoluta), sem a constante -L*N/sigma_s."""
        out, sigma, nu = LocalSensitivityScores._source_terms(
            samples,
            x,
            baseline_source_model
        )
        return out['sigma']
    

    @staticmethod
    def source_df_score(
        samples,
        x,
        baseline_prior,
        baseline_source_model
    ):
        """d/d nu_s ao longo de sigma_s(nu_s) = sqrt((nu_s-2)/nu_s)."""
        out, sigma, nu = LocalSensitivityScores._source_terms(
            samples,
            x,
            baseline_source_model
        )
        return out['nu_partial'] + out['sigma'] * LocalSensitivityScores._unit_variance_dsigma_dnu(sigma, nu)


class GenerateConfigs:

    @staticmethod
    def get_discrete_configs(
        mode,
        n_realizations,
        base_output_path,
        experiment_name,
        experiment_dir,
        create_folder_structure,
        nsources,
        nobs,
        A,
        B,
        seed
    ): 
        def __get_general_configs(
            mode,
            n_realizations,
            base_output_path,
            experiment_name,
            experiment_dir,
            create_folder_structure,
            nsources,
            nobs,
            A,
            B,
            seed
        ):
            
            if mode=='narrow':
                __n_realizations=n_realizations
            elif mode=='broad':
                __n_realizations=1
            else:
                raise ValueError("mode argument must be either 'narrow' or 'broad'.")

            return {
                # General output directory
                'base_output_path': base_output_path,
                # Experiment name for folder structure
                'experiment_name': experiment_name,
                # Experiment directory
                'experiment_dir': experiment_dir,
                # Whether or not to create folder structure
                'create_folder_structure': create_folder_structure,
                # Whether or not to save graphs
                'save_graphs': False,
                # Number of sources
                'n_sources': nsources,
                # Number of observations in each realization
                'n_obs': nobs,
                # Number of parallel workers
                'n_workers': os.cpu_count(),
                # Number of parallel realizations to run
                'n_realizations': __n_realizations,
                # Mixing matrix
                'A': A,
                # Separating matrix
                'B': B,
                # Initial condition
                'initial_B': B,
                # Whether or not to use a normalized posterior
                'normalize_posterior': True,
                # Whether or not to have a progress bar for each initialization test case
                'initialization_progress_bar': False,
                # Whether or not to have a progress bar for each realization
                'realizations_progress_bar': True,
                # Master seed
                'seed': seed
            }
        
        def __get_initialization_configs(mode):
            if mode=='narrow':
                return {
                    'strategy': 'gmm',
                    'starting_value_experiment': 'CBA_SBRT_broad_analysis'
                }
            elif mode=='broad':
                return {
                    'strategy': 'random',
                    'starting_value_experiment': 'posterior_broad_analysis_low_var_1M'
                }
        

        def __get_contour_configs(mode):

            if mode=='narrow':
                grid_points=401
            elif mode=='broad':
                grid_points=3

            return {
                'perfect_model':{
                    # Number of points used in contour line grids
                    'contour_grid_points': grid_points,
                    # Exploration limits for grids
                    'u_lims':(-0.4, 0.4),
                    'v_lims':(-0.4, 0.4),
                    # Central point
                    'central_point': (0,0)
                },
                'slightly_misspecified_model':{
                    # Number of points used in contour line grids
                    'contour_grid_points': grid_points,
                    # Exploration limits for grids
                    'u_lims':(-0.4, 0.4),
                    'v_lims':(-0.4, 0.4),
                    # Central point
                    'central_point': (0,0)
                },
                'largely_misspecified_model':{
                    # Number of points used in contour line grids
                    'contour_grid_points': grid_points,
                    # Exploration limits for grids
                    'u_lims':(-0.8, 0.0),
                    'v_lims':(0.7, 1.5),
                    # Central point
                    'central_point': (-0.4,1.1)
                },
            }

        def __get_map_configs(mode):

            if mode=='narrow':
                max_it=1500000
                min_it=1000
            elif mode=='broad':
                max_it=2
                min_it=1
            
            return {
                # Threshold for stopping optimization
                'stopping_thresh': 1E-11,
                # Maximum number of iterations
                'max_it': max_it,
                # Minimum iterations
                'min_it': min_it,
                # Learning rate
                'learning_rate': 1E-6,
                # Persistance iterations for stopping criterion
                'stopping_criterion_persistance_its': 20
            }
        
        def __get_mcmc_configs(mode):

            if mode=='narrow':
                scale=0.1
                exploration_var=5E-4
                n_samples=30000
                parallel_chains=4
                max_it=20000
                R_hat_minimum_burn_in=2000
                auto_adjust_exploration_var=True
            elif mode=='broad':
                scale=10
                exploration_var=1E-4
                n_samples=100000
                parallel_chains=100
                max_it=25000
                R_hat_minimum_burn_in=10000
                auto_adjust_exploration_var=False
            
            return {
                # Initial condition noise
                'starting_distribution': lambda: st.t.rvs(
                    loc=0,
                    scale=scale,
                    df=3,
                    size=B.shape
                ),
                # Exploration variance for MCMC
                'exploration_var': exploration_var,
                # Number of samples to generate
                'n_samples': n_samples,
                # Number of parallel chains for MCMC
                'parallel_chains': parallel_chains,
                # Maximum iterations for MCMC
                'max_it':max_it,
                # Convergence threshold for R-hat diagnostic
                'R_hat_thresh': 1.01,
                # Interval at which R-hat will be evaluated
                'R_hat_evaluation_step': 500,
                # Number of consecutive convergent evaluations necessary
                'R_hat_persistance': 2,
                # Samples which will be discarded
                'R_hat_minimum_burn_in': R_hat_minimum_burn_in,
                # Whether or not to adjust exploration variance inside MCMC
                'auto_adjust_exploration_var': auto_adjust_exploration_var,
                # Whether or not to show progress bars
                'progress_bar': False,
                # Target acceptance probability, in case auto adjust is enabled
                'target_accept_prob': 0.5,

            }
        
        def __get_sim_configs(
            B,
            nsources
        ):
            return {
                'source_model': StandardLogisticSource(use_jax=True),
                'sources': {
                    'perfect_model': StandardLogisticSource(use_jax=True),
                    'slightly_misspecified_model': LogisticSource(
                        mu=0.1,
                        sigma=1.1,
                        use_jax=True
                    ),
                    'largely_misspecified_model': TriangularSource(
                        lower=-2,
                        upper=2,
                        mode=0,
                        use_jax=True
                    )
                },
                'priors': {
                    'likelihood': ExponentialPrior(
                        center=B,
                        std=np.inf,
                        use_jax=True
                    ),
                    'non_informative_prior': ExponentialPrior(
                        center=B,
                        std=10,
                        use_jax=True
                    ),
                    'informative_prior': ExponentialPrior(
                        center=B,
                        std=0.1,
                        use_jax=True
                    ),
                    'identity_transform': ExponentialPrior(
                        center=np.eye(nsources),
                        std=0.1,
                        use_jax=True
                    )
                },
                'test_cases': {
                    'i': {
                        'source': 'perfect_model',
                        'prior': 'likelihood',
                    },
                    'ii': {
                        'source': 'perfect_model',
                        'prior': 'non_informative_prior',
                    },
                    'iii': {
                        'source': 'perfect_model',
                        'prior': 'informative_prior',
                    },
                    'iv': {
                        'source': 'perfect_model',
                        'prior': 'identity_transform',
                    },
                    'v': {
                        'source': 'slightly_misspecified_model',
                        'prior': 'likelihood',
                    },
                    'vi': {
                        'source': 'slightly_misspecified_model',
                        'prior': 'non_informative_prior',
                    },
                    'vii': {
                        'source': 'slightly_misspecified_model',
                        'prior': 'informative_prior',
                    },
                    'viii': {
                        'source': 'slightly_misspecified_model',
                        'prior': 'identity_transform',
                    },
                    'ix': {
                        'source': 'largely_misspecified_model',
                        'prior': 'likelihood',
                    },
                    'x': {
                        'source': 'largely_misspecified_model',
                        'prior': 'non_informative_prior',
                    },
                    'xi': {
                        'source': 'largely_misspecified_model',
                        'prior': 'informative_prior',
                    },
                    'xii': {
                        'source': 'largely_misspecified_model',
                        'prior': 'identity_transform',
                    }
                }
            }
        

        configs = {}

        # Add general configs
        configs['general'] = __get_general_configs(
            mode=mode,
            n_realizations=n_realizations,
            base_output_path=base_output_path,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            create_folder_structure=create_folder_structure,
            nsources=nsources,
            nobs=nobs,
            A=A,
            B=B,
            seed=seed
        )

        # Add initialization configs
        configs['initialization'] = __get_initialization_configs(mode=mode)

        # Add contour configs
        configs['contour'] = __get_contour_configs(mode=mode)

        # Add MAP configs
        configs['map'] = __get_map_configs(mode=mode)

        # Add MCMC configs
        configs['mcmc'] = __get_mcmc_configs(mode=mode)

        # Add simulation configs
        configs['sim'] = __get_sim_configs(
            B=B,
            nsources=nsources
        )

        return configs


    @staticmethod
    def get_continuous_configs(
        n_realizations,
        base_output_path,
        experiment_name,
        experiment_dir,
        create_folder_structure,
        nsources,
        nobs,
        A,
        B,
        source_dfs,
        seed
    ):
        def __get_general_configs(
            n_realizations,
            base_output_path,
            experiment_name,
            experiment_dir,
            create_folder_structure,
            nsources,
            nobs,
            A,
            B,
            seed
        ):
            return {
                # General output directory
                'base_output_path': base_output_path,
                # Experiment name for folder structure
                'experiment_name': experiment_name,
                # Experiment directory
                'experiment_dir': experiment_dir,
                # Whether or not to create folder structure
                'create_folder_structure': create_folder_structure,
                # Whether or not to save graphs
                'save_graphs': False,
                # Number of sources
                'n_sources': nsources,
                # Number of observations in each realization
                'n_obs': nobs,
                # Number of parallel workers
                'n_workers': os.cpu_count(),
                # Number of parallel realizations to run
                'n_realizations': n_realizations,
                # Mixing matrix
                'A': A,
                # Separating matrix
                'B': B,
                # Initial condition
                'initial_B': B,
                # Whether or not to use a normalized posterior
                'normalize_posterior': True,
                # Whether or not to have a progress bar for each initialization test case
                'initialization_progress_bar': False,
                # Whether or not to have a progress bar for each realization
                'realizations_progress_bar': True,
                # Master seed
                'seed': seed
            }
        
        def __get_prior_variation_configs(
            nsources,
            B
        ):
            return {
                'mode':{
                    'base_prior': MultivariateTPrior(
                        center=0.5*(B + np.eye(nsources)),
                        scale=np.sqrt(1/3),
                        df=3,
                        use_jax=True
                    ),
                    'target_param_values': np.linspace(
                        start=0,
                        stop=1,
                        num=25
                    ),
                    'target_param_fn': lambda target_params: [
                        MultivariateTPrior(
                            center=(1-p)*B + p*np.eye(nsources),
                            scale=np.sqrt(1/3),
                            df=3,
                            use_jax=True
                        ) for p in target_params
                    ],
                    'score_fn': LocalSensitivityScores.prior_mode_score(
                        direction=np.eye(nsources)-B
                    )
                },
                'scale':{
                    'base_prior': MultivariateTPrior(
                        center=B,
                        scale=1E-1,
                        df=3,
                        use_jax=True
                    ),
                    'target_param_values': np.logspace(
                        start=np.log10(0.05),
                        stop=0,
                        num=25,
                        base=10
                    ),
                    'target_param_fn': lambda target_params: [
                        MultivariateTPrior(
                            center=B,
                            scale=p,
                            df=3,
                            use_jax=True
                        ) for p in target_params
                    ],
                    'score_fn': LocalSensitivityScores.prior_scale_score
                },
                'tail_heaviness': {
                    'base_prior': MultivariateTPrior(
                        center=B,
                        scale=np.sqrt(1/3),
                        df=3,
                        use_jax=True
                    ),
                    'target_param_values': range(3, 30),
                    'target_param_fn': lambda target_params: [
                        MultivariateTPrior(
                            center=B,
                            scale=np.sqrt((p - 2) / p),
                            df=p,
                            use_jax=True
                        )
                        for p in target_params
                    ],
                    'score_fn': LocalSensitivityScores.prior_df_score
                },
            }
        
        def __get_source_model_variation_configs(
            B
        ):
            return {
                'mode':{
                    'param_name': 'mu',
                    'base_prior': MultivariateTPrior(
                        center=B,
                        scale=np.sqrt(1/3),
                        df=3,
                        use_jax=True
                    ),
                    'target_param_values': np.linspace(
                        start=-0.2,
                        stop=0.2,
                        num=25
                    ),
                    'target_param_fn': lambda baseline_objs, target_params, param_name: [
                        setattr(
                            b_obj,
                            param_name,
                            p
                        ) for b_obj, p in zip(
                            baseline_objs,
                            target_params
                        )
                    ],
                    'score_fn': LocalSensitivityScores.source_mu_score
                },
                'scale':{
                    'param_name': 'sigma',
                    'base_prior': MultivariateTPrior(
                        center=B,
                        scale=np.sqrt(1/3),
                        df=3,
                        use_jax=True
                    ),
                    'target_param_values': np.linspace(
                        0.97,
                        1.03,
                        25
                    ),
                    'target_param_fn': lambda baseline_objs, target_params, param_name: [
                        setattr(
                            b_obj,
                            param_name,
                            b_obj.sigma*p
                        ) for b_obj, p in zip(
                            baseline_objs,
                            target_params
                        )
                    ],
                    'score_fn': LocalSensitivityScores.source_sigma_score
                },
                'tail_heaviness': {
                    'param_name': 'df',
                    'base_prior': MultivariateTPrior(
                        center=B,
                        scale=np.sqrt(1/3),
                        df=3,
                        use_jax=True
                    ),
                    'target_param_values': np.linspace(
                        0.95,
                        1.05,
                        25
                    ),
                    'target_param_fn': lambda baseline_objs, target_params, param_name: [
                        (
                            setattr(
                                b_obj,
                                'df',
                                float(b_obj.df) * r
                            ),                       
                            setattr(
                                b_obj,
                                'sigma',
                                np.sqrt((b_obj.df - 2) / b_obj.df)
                            )      # usa o df novo
                        ) for b_obj, r in zip(baseline_objs, target_params)
                    ],
                    'score_fn': LocalSensitivityScores.source_df_score
                },
            }
            
        def __get_prior_source_model_variation_configs():
            return {}
        

        def __get_mcmc_configs():
            return {
                # Initial condition noise
                'starting_distribution': lambda: st.t.rvs(
                    loc=0,
                    scale=0.05,
                    df=3,
                    size=B.shape
                ),
                # Exploration variance for MCMC
                'exploration_var': 1E-4,
                # Number of samples to generate
                'n_samples': 30000,
                # Number of parallel chains for MCMC
                'parallel_chains': 4,
                # Maximum iterations for MCMC
                'max_it': 20000,
                # Convergence threshold for R-hat diagnostic
                'R_hat_thresh': 1.01,
                # Interval at which R-hat will be evaluated
                'R_hat_evaluation_step': 500,
                # Number of consecutive convergent evaluations necessary
                'R_hat_persistance': 2,
                # Samples which will be discarded
                'R_hat_minimum_burn_in': 5000,
                # Whether or not to adjust exploration variance inside MCMC
                'auto_adjust_exploration_var': True,
                # Whether or not to show progress bars
                'progress_bar': False,
                # Target acceptance probability, in case auto adjust is enabled
                'target_accept_prob': 0.5,
            }
        
        def __get_sources_configs(
            source_dfs
        ):
            return {
                'rs{}'.format(s+1): StudentTSource(
                    mu=0,
                    sigma=np.sqrt((df-2)/df),
                    df=df,
                    use_jax=True
                ) for s, df in enumerate(source_dfs)
            }


        configs = {}

        # Add general configs
        configs['general'] = __get_general_configs(
            n_realizations=n_realizations,
            base_output_path=base_output_path,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            create_folder_structure=create_folder_structure,
            nsources=nsources,
            nobs=nobs,
            A=A,
            B=B,
            seed=seed
        )

        # Add prior variation configs
        configs['prior_variation'] = __get_prior_variation_configs(
            nsources=nsources,
            B=B
        )

        # Add source model variation configs
        configs['source_model_variation'] = __get_source_model_variation_configs(
            B=B
        )

        # Add prior + source model variation configs
        configs['prior_source_model_variation'] = __get_prior_source_model_variation_configs()

        # Add mcmc configs
        configs['mcmc'] = __get_mcmc_configs()

        # Add sources configs
        configs['sources'] = __get_sources_configs(
            source_dfs=source_dfs
        )

        return configs

class SignalGraphPlotter:

    @classmethod
    def plot_sources(
        cls,
        s,
        source_model,
        nsources=2,
        bin_cfg='auto'
    ):
        # TODO: CONFIGURAR BINS AUTOMATICOS ('fd')

        # Get distributions from source model
        source_model_pdf = source_model.get()
        source_model_cumulative = source_model.get_cumulative()
        
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(20,15)
        )
        
        
        for i in range(1, nsources+1):
            ax1.hist(
                x=s[i-1,:],
                bins=100,
                density=True,
                label='s{}'.format(i-1),
                alpha=0.5,
                # bin_cfg='auto'
            )
            ax2.hist(
                x=s[i-1,:],
                bins=100,
                cumulative=True,
                density=True,
                label='s{}'.format(i-1),
                alpha=0.5,
                # bin_cfg='auto'
            )
            
        ax1.plot(
            np.linspace(-10,10,1000),
            [source_model_pdf(x) for x in np.linspace(-10,10,1000)],
            label='sigmoide teórica'
        )
        
        
        ax2.plot(
            np.linspace(-10,10,1000),
            [source_model_cumulative(x) for x in np.linspace(-10,10,1000)],
            label='sigmoide teórica'
        )
        
        ax1.set_xlabel(
            '$s_{i}$',
            fontsize=15
        )
        ax1.set_ylabel(
            '$f(s_{i})$',
            fontsize=15
        )
        ax1.set_title(
            'Densidade de probabilidade das fontes $f(s_{i})$',
            fontsize=20
        )
        
        ax2.set_xlabel(
            '$s_{i}$',
            fontsize=15
        )
        ax2.set_ylabel(
            '$F(s_{i})$',
            fontsize=15
        )
        ax2.set_title(
            'Densidade de probabilidade cumulativa das fontes $F(s_{i})$',
            fontsize=20
        )
        
        l1=ax1.legend(fontsize=15)
        l2=ax2.legend(fontsize=15)

    @classmethod
    def plot_mixtures(
        cls,
        x,
        nsources=2
    ):
        
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(20,15)
        )
        
        for i in range(1, nsources+1):
            ax1.hist(
                x=x[i-1,:],
                bins=100,
                density=True,
                label='x{}'.format(i-1),
                alpha=0.5
            )
            ax2.hist(
                x=x[i-1,:],
                bins=100,
                cumulative=True,
                density=True,
                label='x{}'.format(i-1),
                alpha=0.5
            )
        
        ax1.set_xlabel(
            '$x_{i}$',
            fontsize=15
        )
        ax1.set_ylabel(
            '$f(x_{i})$',
            fontsize=15
        )
        ax1.set_title(
            'Densidades de probabilidade das observações $f(x_{i})$',
            fontsize=20
        )
        
        ax2.set_xlabel(
            '$x_{i}$',
            fontsize=15
        )
        ax2.set_ylabel(
            '$F(x_{i})$',
            fontsize=15
        )
        ax2.set_title(
            'Densidades de probabilidade cumulativa das observações $F(x_{i})$',
            fontsize=20
        )
        
        l1=ax1.legend(fontsize=15)
        l2=ax2.legend(fontsize=15)  

        

class MCMCGraphPlotter:

    @classmethod
    def evolution_of_sampled_coefficients(
        cls,
        estimator,
        samples,
        logs,
        save_dir,
        ylims=None,
        label_size=15,
        tick_size=15,
        legend_size=15
    ):
        print('-'*100)
        print('Evolution of sampled coefficients')
        print('(a) B00')
        print('(b) B01')
        print('(c) B10')
        print('(d) B11')
        print('-'*100)
        
        # Plot sampled coefficients
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        # B_00
        axs[0,0].plot(
            logs.iteration,
            samples[:, 0, 0],
            label='samples'
        )
        axs[0,0].axvline(
            estimator.burn_in_start,
            color='red',
            linestyle='--',
            linewidth=4,
            label='Burn-in threshold'
        )
        axs[0,0].set_xlabel(
            'iteration',
            fontsize=label_size
        )
        axs[0,0].set_ylabel(
            '$b_{11}$',
            fontsize=label_size
        )
        axs[0,0].set_title(
            '(a)',
            fontsize=label_size
        )
        axs[0,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[0,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[0,0].legend(
            fontsize=legend_size
        )
        
        # B_01
        axs[0,1].plot(
            logs.iteration,
            samples[:, 0, 1],
            label='samples'
        )
        axs[0,1].axvline(
            estimator.burn_in_start,
            color='red',
            linestyle='--',
            linewidth=4,
            label='Burn-in threshold'
        )
        axs[0,1].set_xlabel(
            'iteration',
            fontsize=label_size
        )
        axs[0,1].set_ylabel(
            '$b_{12}$',
            fontsize=label_size
        )
        axs[0,1].set_title(
            '(b)',
            fontsize=label_size
        )
        axs[0,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[0,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[0,1].legend(
            fontsize=legend_size
        )
        
        # B_10
        axs[1,0].plot(
            logs.iteration,
            samples[:, 1, 0],
            label='samples'
        )
        axs[1,0].axvline(
            estimator.burn_in_start,
            color='red',
            linestyle='--',
            linewidth=4,
            label='Burn-in threshold'
        )
        axs[1,0].set_xlabel(
            'iteration',
            fontsize=label_size
        )
        axs[1,0].set_ylabel(
            '$B_{21}$',
            fontsize=label_size
        )
        axs[1,0].set_title(
            '(c)',
            fontsize=label_size
        )
        axs[1,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[1,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[1,0].legend(
            fontsize=legend_size
        )
        
        # B_11
        axs[1,1].plot(
            logs.iteration,
            samples[:, 1, 1],
            label='samples'
        )
        axs[1,1].axvline(
            estimator.burn_in_start,
            color='red',
            linestyle='--',
            linewidth=4,
            label='Burn-in threshold'
        )
        axs[1,1].set_xlabel(
            'iteration',
            fontsize=label_size
        )
        axs[1,1].set_ylabel(
            '$B_{22}$',
            fontsize=label_size
        )
        axs[1,1].set_title(
            '(d)',
            fontsize=label_size
        )
        axs[1,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[1,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[1,1].legend(
            fontsize=legend_size
        )

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'evolution_of_sampled_coefficients.png')
                )
            )


    @classmethod
    def evolution_of_samples_distribution(
        cls,
        B_est,
        samples,
        step_size,
        palette,
        save_dir,
        ylims=None,
        xlims=None,
        label_size=15,
        tick_size=15,
        legend_size=15
    ):
        n_samples=len(samples)
        i=0
        evaluated_intervals=[]
        while i<n_samples:
            start=i
            end=min(
                i+step_size,
                n_samples
            )
            evaluated_intervals.append(
                (start, end)
            )
            i = i + step_size
        
        # Window samples and construct dataframe for plotting
        plot_df=pd.DataFrame()
        for start, end in evaluated_intervals:
            wdw_df = pd.DataFrame(
                data={
                    'interval': ['{},{}'.format(start, end)]*(end-start)
                }
            )
            wdw_samples = samples[start:end,:,:]
            for i, j in np.ndindex(B_est.shape):
                wdw_df['B_{}{}'.format(i, j)] = wdw_samples[:,i,j]
        
            plot_df = pd.concat(
                [
                    plot_df,
                    wdw_df
                ],
                axis=0
            ).reset_index(
                drop=True
            )
        
        print('-'*100)
        print('Evolution of coefficient distributions')
        print('(a) B00')
        print('(b) B01')
        print('(c) B10')
        print('(d) B11')
        print('-'*100)
        
        # Plot coefficient distribution evolution
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        # B_00
        sns.kdeplot(
            data=plot_df,
            x='B_00',
            hue='interval',
            ax=axs[0,0],
            palette=palette
        )
        axs[0,0].set_xlabel(
            '$B_{00}$',
            fontsize=label_size
        )
        axs[0,0].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[0,0].set_title(
            '(a)',
            fontsize=label_size
        )
        axs[0,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[0,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[0,0].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        axs[0,0].legend(
            loc='upper right',
            fontsize=legend_size
        )
        
        # B_01
        sns.kdeplot(
            data=plot_df,
            x='B_01',
            hue='interval',
            ax=axs[0,1],
            palette=palette
        )
        axs[0,1].set_xlabel(
            '$B_{01}$',
            fontsize=label_size
        )
        axs[0,1].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[0,1].set_title(
            '(b)',
            fontsize=label_size
        )
        axs[0,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[0,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[0,1].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        axs[0,1].legend(
            loc='upper right',
            fontsize=legend_size
        )
        
        # B_10
        sns.kdeplot(
            data=plot_df,
            x='B_10',
            hue='interval',
            ax=axs[1,0],
            palette=palette
        )
        axs[1,0].set_xlabel(
            '$B_{10}$',
            fontsize=label_size
        )
        axs[1,0].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[1,0].set_title(
            '(C)',
            fontsize=label_size
        )
        axs[1,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[1,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[1,0].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        axs[1,0].legend(
            loc='upper right',
            fontsize=legend_size
        )
        
        # B_11
        sns.kdeplot(
            data=plot_df,
            x='B_11',
            hue='interval',
            ax=axs[1,1],
            palette=palette
        )
        axs[1,1].set_xlabel(
            '$B_{11}$',
            fontsize=label_size
        )
        axs[1,1].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[1,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[1,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[1,1].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        axs[1,1].legend(
            loc='upper right',
            fontsize=legend_size
        )

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'evolution_of_samples_distribution.png')
                )
            )


    @classmethod
    def steady_state_marginal_distributions(
        cls,
        valid_samples,
        nbins,
        B_est,
        B_true,
        save_dir,
        ylims=None,
        xlims=None,
        label_size=15,
        tick_size=15,
        legend_size=15,
        plot_estimate=True,
        plot_true_value=True
    ):

        print('-'*100)
        print('(a) B11')
        print('(b) B12')
        print('(c) B21')
        print('(d) B22')
        print('-'*100)
        
        # Plot sampled coefficients
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        # B_00
        axs[0,0].hist(
            valid_samples[:, 0, 0],
            density=True,
            bins=nbins,
            label='amostras'
        )
        if plot_true_value:
            axs[0,0].axvline(
                B_true[0, 0],
                color='red',
                linestyle='--',
                linewidth=4,
                label='$b_{11}$'
            )
        if plot_estimate:
            axs[0,0].axvline(
                B_est[0, 0],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label=r'$\hat{b}_{11}$'
            )
            
        axs[0,0].set_xlabel(
            '$b_{11}$',
            fontsize=label_size
        )
        axs[0,0].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[0,0].set_title(
            '(a)',
            fontsize=label_size
        )
        axs[0,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[0,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[0,0].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        if plot_estimate:
            axs[0,0].legend(
                loc='upper right',
                fontsize=legend_size
            )
        
        
        # B_01
        axs[0,1].hist(
            valid_samples[:, 0, 1],
            density=True,
            bins=nbins,
            label='amostras'
        )
        if plot_true_value:
            axs[0,1].axvline(
                B_true[0, 1],
                color='red',
                linestyle='--',
                linewidth=4,
                label='$b_{12}$'
            )
        if plot_estimate:
            axs[0,1].axvline(
                B_est[0, 1],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label=r'$\hat{b}_{12}$'
            )
        axs[0,1].set_xlabel(
            '$b_{12}$',
            fontsize=label_size
        )
        axs[0,1].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[0,1].set_title(
            '(b)',
            fontsize=label_size
        )
        axs[0,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[0,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[0,1].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        if plot_estimate:
            axs[0,1].legend(
                loc='upper right',
                fontsize=legend_size
            )
        
        # B_10
        axs[1,0].hist(
            valid_samples[:, 1, 0],
            density=True,
            bins=nbins,
            label='amostras'
        )
        if plot_true_value:
            axs[1,0].axvline(
                B_true[1, 0],
                color='red',
                linestyle='--',
                linewidth=4,
                label='$b_{21}$'
            )
        if plot_estimate:
            axs[1,0].axvline(
                B_est[1, 0],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label=r'$\hat{b}_{21}$'
            )
        axs[1,0].set_xlabel(
            '$b_{21}$',
            fontsize=label_size
        )
        axs[1,0].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[1,0].set_title(
            '(c)',
            fontsize=label_size
        )
        axs[1,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[1,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[1,0].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        if plot_estimate:
            axs[1,0].legend(
                loc='upper right',
                fontsize=legend_size
            )
        
        # B_11
        axs[1,1].hist(
            valid_samples[:, 1, 1],
            density=True,
            bins=nbins,
            label='amostras'
        )
        if plot_true_value:
            axs[1,1].axvline(
                B_true[1, 1],
                color='red',
                linestyle='--',
                linewidth=4,
                label='$b_{22}$'
            )
        if plot_estimate:
            axs[1,1].axvline(
                B_est[1, 1],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label=r'$\hat{b}_{22}$'
            )
        axs[1,1].set_xlabel(
            '$b_{22}$',
            fontsize=label_size
        )
        axs[1,1].set_ylabel(
            'density',
            fontsize=label_size
        )
        axs[1,1].set_title(
            '(d)',
            fontsize=label_size
        )
        axs[1,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            axs[1,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if xlims is not None:
            axs[1,1].set_xlim(
                left=xlims[0],
                right=xlims[-1]
            )
        if plot_estimate:
            axs[1,1].legend(
                loc='upper right',
                fontsize=legend_size
            )

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'steady_state_marginal_distributions.png')
                )
            )


    @classmethod
    def evolution_log_posterior(
        cls,
        logs,
        save_dir,
        ylims=None,
        label_size=15,
        tick_size=15
    ):
        print('-'*100)
        print('Evolution of log-posterior - MMSE')
        print('-'*100)
        
        fig = plt.figure(figsize=(20,7))
        logs['log_posterior']=logs['log_posterior']/1000
        sns.lineplot(
            data=logs,
            x='iteration',
            y='log_posterior'
        )
        plt.ylabel(
            r'$\frac{1}{N} \log p_{\bf{B}|\mathcal{X}}(\bf{B}|\mathcal{X})$',
            fontsize=label_size
        )
        plt.xlabel(
            'k',
            fontsize=label_size
        )
        plt.xticks(
            fontsize=tick_size
        )
        plt.yticks(
            fontsize=tick_size
        )
        if ylims is not None:
            plt.ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'evolution_log_posterior.png')
                )
            )


    @classmethod
    def source_separation_results(
        cls,
        plot_start,
        plot_end,
        B_est,
        s,
        s_est,
        x,
        nobs,
        save_dir,
        ylims=None,
        label_size=15,
        tick_size=15,
        legend_size=15
    ):
        print('#'*100)
        print('SOURCE SEPARATION RESULTS - MMSE')
        print('-'*100)
        print('(a) Time series - true values and estimates - coefficient s0')
        print('(b) Scatter plot - true values and estimates - coefficient s0')
        print('(c) Time series - true values and estimates - coefficient s1')
        print('(d) Scatter plot - true values and estimates - coefficient s1')
        print('#'*100)
        
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        t=range(nobs)
        
        # Axs 00
        axs[0,0].plot(
            t[plot_start:plot_end],
            s_est[0,plot_start:plot_end],
            label=r'$\hat{s}_{0}$',
            color='red',
            linestyle='--'
        )
        axs[0,0].plot(
            t[plot_start:plot_end],
            s[0,plot_start:plot_end],
            label='$s_{0}$'
        )
        axs[0,0].set_xlabel(
            'n',
            fontsize=label_size
        )
        axs[0,0].set_title(
            '(a)',
            fontsize=label_size
        )
        axs[0,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[0,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[0,0].legend(fontsize=legend_size)
        
        # Axs 01
        axs[0,1].scatter(
            s[0,:],
            s_est[0,:],
        )
        axs[0,1].set_xlabel(
            '$s_{0}$',
            fontsize=label_size
        )
        axs[0,1].set_ylabel(
            r'$\hat{s}_{0}$',
            fontsize=label_size
        )
        axs[0,1].set_title(
            '(b)',
            fontsize=label_size
        )
        axs[0,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[0,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        
        # Axs 10
        axs[1,0].plot(
            t[plot_start:plot_end],
            s_est[1,plot_start:plot_end],
            label=r'$\hat{s}_{1}$',
            color='red',
            linestyle='--'
        )
        axs[1,0].plot(
            t[plot_start:plot_end],
            s[1,plot_start:plot_end],
            label='$s_{1}$'
        )
        axs[1,0].set_xlabel(
            'n',
            fontsize=label_size
        )
        axs[1,0].set_title(
            '(c)',
            fontsize=label_size
        )
        axs[1,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[1,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[1,0].legend(fontsize=legend_size)
        
        # Axs 11
        axs[1,1].scatter(
            s[1,:],
            s_est[1,:],
        )
        axs[1,1].set_xlabel(
            '$s_{1}$',
            fontsize=label_size
        )
        axs[1,1].set_ylabel(
            r'$\hat{s}_{1}$',
            fontsize=label_size
        )
        axs[1,1].set_title(
            '(d)',
            fontsize=label_size
        )
        axs[1,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[1,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'source_separation_results_mmse.png')
                )
            )

class MAPGradientAscentGraphPlotter:

    @classmethod
    def evolution_log_posterior(
        cls,
        logs,
        save_dir,
        ylims=None,
        label_size=15,
        tick_size=15
    ):
        print('-'*100)
        print('Evolution of log-posterior - MAP')
        print('-'*100)
        
        fig = plt.figure(figsize=(20,7))
        sns.lineplot(
            data=logs,
            x='iteration',
            y='log_posterior'
        )
        plt.ylabel(
            r'$\frac{1}{N} \log p_{\bf{B}|\mathcal{X}}(\bf{B}|\mathcal{X})$',
            fontsize=label_size
        )
        plt.xlabel(
            'k',
            fontsize=label_size
        )
        plt.xticks(
            fontsize=tick_size
        )
        plt.yticks(
            fontsize=tick_size
        )
        if ylims is not None:
            plt.ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'evolution_log_posterior.png')
                )
            )

    @classmethod
    def evolution_Best_determinant(
        cls,
        logs,
        save_dir,
        ylims=None,
        label_size=15,
        tick_size=15
    ):
        print('-'*100)
        print('Evolução do determinante de B ao longo das iterações')
        print('-'*100)
        fig = plt.figure(figsize=(20,7))
        sns.lineplot(
            data=logs,
            x='iteration',
            y='detB'
        )
        plt.ylabel(
            r'$\det(\hat{B})$',
            fontsize=label_size
        )
        plt.xlabel(
            'iteration',
            fontsize=label_size
        )
        plt.xticks(
            fontsize=tick_size
        )
        plt.yticks(
            fontsize=tick_size
        )
        if ylims is not None:
            plt.ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'evolution_Best_determinant.png')
                )
            )

    @classmethod
    def evolution_Best_derivative(
        cls,
        logs,
        save_dir,
        B_est,
        ylims=None,
        label_size=15,
        tick_size=15,
        legend_size=15
    ):

        print('-'*100)
        print('Evolução do determinante de B ao longo das iterações')
        print('-'*100)
        
        fig = plt.figure(figsize=(20,7))
        for i, j in np.ndindex(B_est.shape):
            plt.plot(
                logs.iteration.values,
                [row['gradient'][i,j] for n, row in logs.iterrows()],
                label='b{}{}'.format(i,j)
            )
        
        plt.ylabel(
            r'$d \;\log P(\hat{B}|X)/d \; bij$',
            fontsize=label_size
        )
        plt.xlabel(
            'iteration',
            fontsize=label_size
        )
        plt.xticks(
            fontsize=tick_size
        )
        plt.yticks(
            fontsize=tick_size
        )
        plt.legend(
            fontsize=legend_size
        )
        if ylims is not None:
            plt.ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'evolution_Best_derivative.png')
                )
            )

    @classmethod
    def source_separation_results(
        cls,
        plot_start,
        plot_end,
        B_est,
        s,
        s_est,
        x,
        nobs,
        save_dir,
        ylims=None,
        label_size=15,
        tick_size=15,
        legend_size=15
    ):
        print('#'*100)
        print('SOURCE SEPARATION RESULTS - MAP')
        print('-'*100)
        print('(a) Time series - true values and estimates - coefficient s0')
        print('(b) Scatter plot - true values and estimates - coefficient s0')
        print('(c) Time series - true values and estimates - coefficient s1')
        print('(d) Scatter plot - true values and estimates - coefficient s1')
        print('#'*100)
        
        
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        t=range(nobs)
        
        # Axs 00
        axs[0,0].plot(
            t[plot_start:plot_end],
            s_est[0,plot_start:plot_end],
            label=r'$\hat{s}_{0}$',
            color='red',
            linestyle='--'
        )
        axs[0,0].plot(
            t[plot_start:plot_end],
            s[0,plot_start:plot_end],
            label='$s_{0}$'
        )
        axs[0,0].set_xlabel(
            'n',
            fontsize=label_size
        )
        axs[0,0].set_title(
            '(a)',
            fontsize=label_size
        )
        axs[0,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[0,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[0,0].legend(fontsize=legend_size)
        
        # Axs 01
        axs[0,1].scatter(
            s[0,:],
            s_est[0,:],
        )
        axs[0,1].set_xlabel(
            '$s_{0}$',
            fontsize=label_size
        )
        axs[0,1].set_ylabel(
            r'$\hat{s}_{0}$',
            fontsize=label_size
        )
        axs[0,1].set_title(
            '(b)',
            fontsize=label_size
        )
        axs[0,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[0,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        
        # Axs 10
        axs[1,0].plot(
            t[plot_start:plot_end],
            s_est[1,plot_start:plot_end],
            label=r'$\hat{s}_{1}$',
            color='red',
            linestyle='--'
        )
        axs[1,0].plot(
            t[plot_start:plot_end],
            s[1,plot_start:plot_end],
            label='$s_{1}$'
        )
        axs[1,0].set_xlabel(
            'n',
            fontsize=label_size
        )
        axs[1,0].set_title(
            '(c)',
            fontsize=label_size
        )
        axs[1,0].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[1,0].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        axs[1,0].legend(fontsize=legend_size)
        
        # Axs 11
        axs[1,1].scatter(
            s[1,:],
            s_est[1,:],
        )
        axs[1,1].set_xlabel(
            '$s_{1}$',
            fontsize=label_size
        )
        axs[1,1].set_ylabel(
            r'$\hat{s}_{1}$',
            fontsize=label_size
        )
        axs[1,1].set_title(
            '(d)',
            fontsize=label_size
        )
        axs[1,1].tick_params(
            axis='both',
            labelsize=tick_size
        )
        if ylims is not None:
            ax[1,1].set_ylim(
                bottom=ylims[0],
                top=ylims[-1]
            )
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'source_separation_results_map.png')
                )
            )

class ContourLineGraphPlotter:

    @classmethod
    def plot_2D_contours(
        cls,
        u_vec,
        v_vec,
        z,
        max_post_point,
        save_dir=None,
        label_size=30,
        tick_size=30,
    ):

        print('-'*100)
        print('Normalized log-posterior (contour lines)')
        print('-'*100)
        
        fig = plt.figure(figsize=(20,7))

        U, V = np.meshgrid(u_vec, v_vec)
        
        plt.contour(
            U,
            V,
            z.T,
            levels=50,
            cmap='copper'
        )
        
        plt.scatter(
            u_vec[max_post_point[0]],
            v_vec[max_post_point[-1]],
            marker='X',
            color='red'
        )
        
        plt.xlabel(
            'u',
            fontsize=label_size
        )
        plt.ylabel(
            'v',
            fontsize=label_size
        )

        plt.xticks(
            fontsize=tick_size
        )
        plt.yticks(
            fontsize=tick_size
        )

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'contour_lines.png')
                )
            )

    @classmethod
    def plot_3D_interactive(
        cls,
        u_vec,
        v_vec,
        z,
        width=1000,
        height=500
    ):
        # Plot interativo
        fig = go.Figure(
            go.Surface(
                x=u_vec,
                y=v_vec,
                z=z
            )
        )
        
        fig.update_layout(
            # title='Log-posteriori normalizada (superfície)',
            autosize=False,
            width=width, height=height,
            scene=dict(
                xaxis_title='U',
                yaxis_title='V',
                zaxis_title='Log-posteriori',
            )
        )
                         
        fig.show()


class EstimationGraphPlotter:
    
    @classmethod
    def estimation_distributions(
        cls,
        B_true,
        estimates,
        nbins,
        save_dir,
        ylims=None,
        xlims=None,
        label_size=15,
        tick_size=15,
        legend_size=15,
        plot_estimate=True,
        plot_true_value=True
    ):

        print('-'*100)
        print('(a) B11')
        print('(b) B12')
        print('(c) B21')
        print('(d) B22')
        print('-'*100)

        # Get true value shape
        n_sources = B_true.shape[0]
        
        # Plot sampled coefficients
        fig, axs = plt.subplots(
            nrows=n_sources, ncols=n_sources,
            figsize=(25,9)
        )

        # Iterate through position and generate plot
        for i, j in np.ndindex(
            (n_sources, n_sources)
        ):
            # Histogram
            axs[i,j].hist(
                estimates[i,j],
                density=True,
                bins=nbins,
                label='estimativas'
            )
            if plot_true_value:
                axs[i,j].axvline(
                    B_true[i, j],
                    color='red',
                    linestyle='--',
                    linewidth=4,
                    label='b_{}{}'.format(i, j)
                )
            if plot_estimate:
                axs[i,j].axvline(
                    np.mean(estimates[i,j]),
                    color='limegreen',
                    linestyle='--',
                    linewidth=4,
                    label='media estimativas'.format(i,j)
                )
                
            axs[i,j].set_xlabel(
                'b_{}{}'.format(i, j),
                fontsize=label_size
            )
            axs[i,j].set_ylabel(
                'density',
                fontsize=label_size
            )

            axs[i,j].tick_params(
                axis='both',
                labelsize=tick_size
            )
            if ylims is not None:
                axs[i,j].set_ylim(
                    bottom=ylims[0],
                    top=ylims[-1]
                )
            if xlims is not None:
                axs[i,j].set_xlim(
                    left=xlims[0],
                    right=xlims[-1]
                )
            if plot_estimate:
                axs[i,j].legend(
                    loc='upper right',
                    fontsize=legend_size
                )

    @classmethod
    def plot_estimation_errors(
        cls,
        errors,
        nbins=20
    ):
        # Get error stats
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)

        # Print stats
        print('#'*100)
        print('Mean error: {}'.format(round(mean_error,2)))
        print('Max error: {}'.format(round(max_error,2)))
        print('Min error: {}'.format(round(min_error,2)))
        print('#'*100)
        
        # Plot
        fig = plt.figure(figsize=(20,7))
        plt.hist(
            errors,
            density=True,
            bins=nbins,
            label='errors'
        )
        plt.axvline(
            np.mean(errors),
            color='limegreen',
            linestyle='--',
            linewidth=4,
            label='media erros'
        )
        plt.legend(
            loc='upper right',
            fontsize=15
        )
        
    

    @classmethod
    def plot_2D_contours(
        cls,
        u_vec,
        v_vec,
        z,
        max_us,
        max_vs,
        N=None,
        posterior_step=0.1,
        levels=200,
        u_lims=(-0.4, 0.4),
        v_lims=(-0.4, 0.4),
        save_dir=None,
        save_name=None,
        label_size=30,
        tick_size=30,
    ):
        
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = 'lightgray'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.5

        print('-'*100)
        print('Normalized log-posterior (average)')
        print('-'*100)

        # Normalize posterior
        if N is not None:
            z = z/N

        # Obtain max and min values for contours
        max_post = np.max([
            x for x in z.flatten() if x < np.inf
        ])
        min_post = np.min([
            x for x in z.flatten() if x > -np.inf
        ])

        # Calculates contour levels based on specified posterior step
        contour_level = max_post
        levels = []
        while contour_level > min_post:
            contour_level -= posterior_step
            levels.append(contour_level)

        fig, ax = plt.subplots(
            nrows=1, ncols=1, 
            figsize=(20,7)
        )

        U, V = np.meshgrid(u_vec, v_vec)
        # import pdb; pdb.set_trace()
        plt.contour(
            U,
            V,
            z.T,
            levels=np.flip(levels),
            cmap='copper',
            label='curva media'
        )
        
        plt.scatter(
            max_us,
            max_vs,
            marker='X',
            color='red',
            label='pontos max'
        )
        
        plt.xlabel(
            'u',
            fontsize=label_size
        )
        plt.ylabel(
            'v',
            fontsize=label_size
        )
        if u_lims is not None:
            plt.xlim(u_lims)
        if v_lims is not None:
            plt.ylim(v_lims)
        
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_color('black')

        plt.xticks(
            np.append(
                np.arange(u_lims[0], u_lims[-1],0.1),
                u_lims[-1]
            ),
            fontsize=tick_size
        )
        plt.yticks(
            np.append(
                np.arange(v_lims[0], v_lims[-1],0.1),
                v_lims[-1]
            ),
            fontsize=tick_size
        )
        # plt.legend(
        #     loc='upper right',
        #     fontsize=15
        # )

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'contour'))
                ),
                bbox_inches='tight'
            )

        # if save_dir is not None:
        #     fig.savefig(
        #         str(
        #             (save_dir / 'contour_lines.png')
        #         )
        #     )
        
        # if save_dir is not None:
        #     fig.savefig(
        #         str(
        #             (save_dir / 'steady_state_marginal_distributions.png')


    @classmethod
    def plot_test_cases_contours(
        cls,
        parser,
        test_cases=None,
        u_range=(-0.3, 0.3),
        v_range=(-0.3, 0.3),
        equal_aspect=False,
        marker_size=40,
        return_stats=True,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
        transparency=0.2,
        legend_location=None
    ):
            
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['savefig.facecolor'] = 'white'
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.color'] = 'lightgray'
            plt.rcParams['grid.linestyle'] = '--'
            plt.rcParams['grid.linewidth'] = 0.5


            # Filter out desired test cases, if so specified
            parsed_results = parser.parsed_results
            
            if test_cases is not None:
                parsed_results = {
                    k: v for k, v in parser.parsed_results.items() if k in test_cases
                }

            # Create dataframe for plotting
            plot_df = pd.DataFrame()
            for test_case, test_case_results in parsed_results.items():
                # Retrieve u coordinate for maximums
                u_vec = [
                    u for u,v in test_case_results['posteriori_grid']['maximums']
                ]

                # Retrieve v coordinate for maximums
                v_vec = [
                    v for u,v in test_case_results['posteriori_grid']['maximums']
                ]

                # Append dataframe
                plot_df = pd.concat(
                    [
                        plot_df,
                        pd.DataFrame(
                            data={
                                'test_case': [test_case]*len(test_case_results['posteriori_grid']['maximums']),
                                'u': u_vec,
                                'v': v_vec
                            }
                        )
                    ],
                    axis=0
                ).reset_index(
                    drop=True
                )

            # Get stats
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                u_mean=('u', 'mean'),
                v_mean=('v', 'mean'),
                u_std=('u', 'std'),
                v_std=('v', 'std')
            )


            # Create plot
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(20,7)
            )
            if equal_aspect:
                ax.set_aspect('equal')

            sns.scatterplot(
                data=plot_df.rename(
                    columns={
                        'test_case': 'Caso de Teste'
                    }
                ),
                x='u',
                y='v',
                hue='Caso de Teste',
                style='Caso de Teste',
                alpha=transparency,
                markers = ['s', '>', '<','o'],
                legend=False,
                **{
                    's': marker_size
                }
            )
            sns.scatterplot(
                data=stats_df.rename(
                    columns={
                        'test_case': 'Caso de Teste',
                        'u_mean': 'u',
                        'v_mean':'v'
                    }
                ),
                x='u',
                y='v',
                hue='Caso de Teste',
                style='Caso de Teste',
                markers = ['s', '>', '<','o'],
                **{
                    's': 3*marker_size
                }
            )

            for spine in ['bottom', 'top', 'left', 'right']:
                ax.spines[spine].set_color('black')

            if legend_location is not None:
                sns.move_legend(ax, legend_location)
            
            plt.xlim(u_range)
            plt.ylim(v_range)
            plt.ylabel('v',fontsize=label_size)
            plt.xlabel('u', fontsize=label_size)
            plt.xticks(fontsize=label_size)
            plt.yticks(fontsize=label_size)
            
            plt.setp(
                ax.get_legend().get_texts(),
                fontsize=legend_size
            )
            plt.setp(
                ax.get_legend().get_title(),
                fontsize=legend_size
            )
            

            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'contours'))
                    ),
                    bbox_inches='tight'
                )

            # Calculate statistics, if so specified
            if return_stats:
                return stats_df

    def plot_map_mmse_errors_boxplots(
        parser,
        test_cases=None,
        B_true=None,
        return_stats=True,
        yranges = None,
        whis_lims=(5,95),
        box_color='gray',
        box_linecolor='black',
        box_linewidth=.5,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
    ):
            
        plt.rc(
            'legend',
            fontsize=legend_size,
            title_fontsize=legend_size
        )

        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'

        plt.rc('axes', labelsize=label_size)
        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)

        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }

        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for estimate_type in ['map', 'mmse']:
            for test_case, test_case_results in parsed_results.items():

                # Create estimate dataframe
                plot_df = pd.concat(
                    [
                        plot_df,
                        pd.DataFrame(
                            data={
                                'estimate_type': [estimate_type]*len(test_case_results[estimate_type]['estimates']),
                                'test_case': [test_case]*len(test_case_results[estimate_type]['estimates']),
                                'b11': test_case_results[estimate_type]['b11_estimates'],
                                'b12': test_case_results[estimate_type]['b12_estimates'],
                                'b21': test_case_results[estimate_type]['b21_estimates'],
                                'b22': test_case_results[estimate_type]['b22_estimates']
                            }
                        )
                    ],
                    axis=0
                ).reset_index(
                    drop=True
                )



        # Plot B_true, if so specified
        if B_true is not None:
            for i,j in np.ndindex(B_true.shape):
                plot_df['b{}{}'.format(i+1, j+1)] = (plot_df['b{}{}'.format(i+1, j+1)] - B_true[i,j])/np.abs(B_true[i,j])

        # Calculate stats_df
        stats_df = plot_df.groupby(
            by=['estimate_type','test_case'],
            as_index=False
        ).agg(
            b11_mean=('b11', 'mean'),
            b12_mean=('b12', 'mean'),
            b21_mean=('b21', 'mean'),
            b22_mean=('b22', 'mean'),
            b11_std=('b11', 'std'),
            b12_std=('b12', 'std'),
            b21_std=('b21', 'std'),
            b22_std=('b22', 'std')
        )

        # Create plot
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(20,12)
        )

        # Create plots in axes
        for i,j in np.ndindex((2,2)):
            # Get coefficient
            coeff = 'b{}{}'.format(i+1,j+1)
            coeff_pos = coeff[1:]

            # Plot horizontal line at 0
            axs[i,j].axhline(
                0,
                color='lightgray',
                linestyle='--'
            )

            sns.boxplot(
                data=plot_df.rename(
                    columns={'estimate_type': 'Tipo de Estimativa'}
                ),
                x='test_case',
                y=coeff,
                hue='Tipo de Estimativa',
                # color=box_color,
                # linecolor=box_linecolor,
                # linewidth=box_linewidth,
                whis=whis_lims,
                width=.7,
                showfliers=False,
                ax=axs[i,j]
            )
            axs[i,j].set_ylabel(
                # r'$\widehat{{b}}_{}-\overline{{b}}_{}$/\overline{{b}}_{}'.format(
                #     '{' + coeff_pos + '}',
                #     '{' + coeff_pos + '}',
                #     '{' + coeff_pos + '}'
                # )
                r'$\delta_{{b}}(\widehat{{b}}_{})$'.format(
                    '{' + coeff_pos + '}',
                    # '{' + coeff_pos + '}',
                    # '{' + coeff_pos + '}'
                )
            )
            for spine in ['bottom', 'top', 'left', 'right']:
                axs[i,j].spines[spine].set_color('black')
            
            axs[i,j].set_xlabel(
                'Caso de Teste'
            )
            # axs[i,j].grid(
            #     visible=True,
            #     axis='y',
            #     **{
            #         'color': 'black',
            #         'linestyle': '--'
            #     }
            # )
            if yranges is not None:
                axs[i,j].set_ylim(
                    yranges[coeff][0],
                    yranges[coeff][-1],
                )
            
            # Plot vertical lines
            [
                axs[i,j].axvline(
                    x+.5,
                    color='lightgray',
                    linestyle='--'
                ) for x in axs[i,j].get_xticks()
            ]

            # Figure legend
            if coeff_pos=='11':
                handles, labels = axs[i,j].get_legend_handles_labels()
                # Create a figure-level legend
                fig.legend(
                    handles,
                    labels,
                    loc='upper center',
                    ncol=3,
                    bbox_to_anchor=(0.5, 0.925)
                )

            # Remove axes legend
            axs[i,j].get_legend().set_visible(False)

            
        
        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'coeffs_estimates_boxplots'))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df
        
    @classmethod
    def plot_coefficient_estimates_boxplots(
        cls,
        parser,
        estimate_type,
        test_cases=None,
        B_true=None,
        return_stats=True,
        xranges = None,
        whis_lims=(5,95),
        box_color='gray',
        box_linecolor='black',
        box_linewidth=.5,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
    ):
            
        plt.rc(
            'legend',
            fontsize=legend_size,
            title_fontsize=legend_size
        )

        plt.rc('axes', labelsize=label_size)
        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)

        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }

        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():

            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(test_case_results[estimate_type]['estimates']),
                            'b11': test_case_results[estimate_type]['b11_estimates'],
                            'b12': test_case_results[estimate_type]['b12_estimates'],
                            'b21': test_case_results[estimate_type]['b21_estimates'],
                            'b22': test_case_results[estimate_type]['b22_estimates'],
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )

        # Plot B_true, if so specified
        if B_true is not None:
            for i,j in np.ndindex(B_true.shape):
                plot_df['b{}{}'.format(i+1, j+1)] = plot_df['b{}{}'.format(i+1, j+1)] - B_true[i,j]

        # Calculate stats_df
        stats_df = plot_df.groupby(
            by='test_case',
            as_index=False
        ).agg(
            b11_mean=('b11', 'mean'),
            b12_mean=('b12', 'mean'),
            b21_mean=('b21', 'mean'),
            b22_mean=('b22', 'mean'),
            b11_std=('b11', 'std'),
            b12_std=('b12', 'std'),
            b21_std=('b21', 'std'),
            b22_std=('b22', 'std')
        )

        # Create plot
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(20,12)
        )

        # Create plots in axes
        for i,j in np.ndindex((2,2)):
            # Get coefficient
            coeff = 'b{}{}'.format(i+1,j+1)
            coeff_pos = coeff[1:]

            sns.boxplot(
                data=plot_df,
                x=coeff,
                y='test_case',
                color=box_color,
                linecolor=box_linecolor,
                linewidth=box_linewidth,
                whis=whis_lims,
                width=.7,
                showfliers=False,
                ax=axs[i,j]
            )
            axs[i,j].set_xlabel(
                r'$\widehat{{b}}_{}^{}-\overline{{b}}_{}$'.format(
                    '{' + coeff_pos + '}',
                    '{' + estimate_type.upper() + '}',
                    '{' + coeff_pos + '}',
                )
            )
            axs[i,j].set_ylabel(
                'Caso de Teste'
            )
            if xranges is not None:
                axs[i,j].set_xlim(
                    xranges[coeff][0],
                    xranges[coeff][-1],
                )
        
        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'coeffs_estimates_boxplots'))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df


        

    @classmethod
    def plot_test_cases_mmse_coefficient_estimates(
        cls,
        parser,
        test_cases=None,
        B_true=None,
        bin_cfg='auto',
        return_stats=True,
        # u_range=(-0.3, 0.3),
        # v_range=(-0.3, 0.3),
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        
        plt.rc(
            'legend',
            fontsize=legend_size,
            title_fontsize=legend_size
        )

        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }
    
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():

            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(test_case_results['mmse']['estimates']),
                            'b11': test_case_results['mmse']['b11_estimates'],
                            'b12': test_case_results['mmse']['b12_estimates'],
                            'b21': test_case_results['mmse']['b21_estimates'],
                            'b22': test_case_results['mmse']['b22_estimates'],
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )
    
        # Create plot
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(20,15)
        )

        # fig.suptitle(
        #     'Casos de teste: {}'.format(', '.join(test_cases)),
        #     fontsize=25
        # )

        # Plot B_true, if so specified
        if B_true is not None:
            for i,j in np.ndindex(B_true.shape):
                axs[i,j].axvline(
                    B_true[i,j],
                    color='red',
                    linestyle='--'
                )

        # B11
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b11',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[0,0]
        )
        axs[0,0].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[0,0].set_xlabel(r'$\widehat{b}_{11}^{MMSE}$',fontsize=label_size)
        axs[0,0].tick_params(
            axis='both',
            labelsize=label_size
        )
        # axs[0,0].legend(
        #     fontsize=legend_size
        # )

        # B12
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b12',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[0,1]
        )
        axs[0,1].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[0,1].set_xlabel(r'$\widehat{b}_{12}^{MMSE}$',fontsize=label_size)
        axs[0,1].tick_params(
            axis='both',
            labelsize=label_size
        )
        # axs[0,1].legend(
        #     fontsize=legend_size
        # )

        # B21
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b21',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[1,0]
        )
        axs[1,0].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[1,0].set_xlabel(r'$\widehat{b}_{21}^{MMSE}$',fontsize=label_size)
        axs[1,0].tick_params(
            axis='both',
            labelsize=label_size
        )
        # axs[1,0].legend(
        #     fontsize=legend_size
        # )

        # B22
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b22',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[1,1]
        )
        axs[1,1].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[1,1].set_xlabel(r'$\widehat{b}_{22}^{MMSE}$',fontsize=label_size)
        axs[1,1].tick_params(
            axis='both',
            labelsize=label_size
        )
        # axs[1,1].legend(
        #     fontsize=legend_size
        # )

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'mmse_coeffs'))
                ),
                bbox_inches='tight'
            )

        # Calculate statistics, if so specified
        if return_stats:
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                b11_mean=('b11', 'mean'),
                b12_mean=('b12', 'mean'),
                b21_mean=('b21', 'mean'),
                b22_mean=('b22', 'mean'),
                b11_std=('b11', 'std'),
                b12_std=('b12', 'std'),
                b21_std=('b21', 'std'),
                b22_std=('b22', 'std')
            )
            return stats_df

    @classmethod
    def plot_test_cases_mmse_errors(
        cls,
        parser,
        test_cases=None,
        bin_cfg='fd',
        return_stats=True,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }
    
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
    
            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(test_case_results['posteriori_grid']['maximums']),
                            'errors': test_case_results['mmse']['errors']
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )
    
        # Create plot
        fig = plt.figure(figsize=(20,10))
        ax = sns.histplot(
            data=plot_df.rename(
                columns={'test_case':'Caso de Teste'}
            ),
            x='errors',
            hue='Caso de Teste',
            bins=bin_cfg
        )
        
        plt.ylabel('Contagem de Realizações', fontsize=label_size)
        plt.xlabel(r'$\delta_{\bf{B}}({\bf{B}})$', fontsize=label_size)
        plt.xticks(fontsize=label_size)
        plt.yticks(fontsize=label_size)
        plt.setp(
            ax.get_legend().get_texts(),
            fontsize=legend_size
        )
        plt.setp(
            ax.get_legend().get_title(),
            fontsize=legend_size
        )

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'mmse_errors'))
                ),
                # bbox_inches='tight'
            )

        # Calculate statistics, if so specified
        if return_stats:
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                errors_mean=('errors', 'mean'),
                errors_min=('errors', 'min'),
                errors_max=('errors', 'max'),
                errors_std=('errors', 'std')
            )
            return stats_df
        
    # @classmethod
    # def plot_test_cases_map_coefficient_estimates(
    #     cls,
    #     parser,
    #     test_cases=None,
    #     B_true=None,
    #     bin_cfg='auto',
    #     return_stats=True,
    #     save_dir=None,
    #     save_name=None,
    #     label_size=25,
    #     legend_size=25
    #     # u_range=(-0.3, 0.3),
    #     # v_range=(-0.3, 0.3),

    # ):
    #     # Filter out desired test cases, if so specified
    #     parsed_results = parser.parsed_results
        
    #     if test_cases is not None:
    #         parsed_results = {
    #             k: v for k, v in parser.parsed_results.items() if k in test_cases
    #         }
    
    #     # Create dataframe for plotting
    #     plot_df = pd.DataFrame()
    #     for test_case, test_case_results in parsed_results.items():

    #         # Append dataframe
    #         plot_df = pd.concat(
    #             [
    #                 plot_df,
    #                 pd.DataFrame(
    #                     data={
    #                         'test_case': [test_case]*len(test_case_results['map']['estimates']),
    #                         'b11': test_case_results['map']['b11_estimates'],
    #                         'b12': test_case_results['map']['b12_estimates'],
    #                         'b21': test_case_results['map']['b21_estimates'],
    #                         'b22': test_case_results['map']['b22_estimates'],
    #                     }
    #                 )
    #             ],
    #             axis=0
    #         ).reset_index(
    #             drop=True
    #         )
    
    #     # Plot figures
    #     for i,j in np.ndindex((2,2)):

    #         fig = plt.figure(figsize=(10,10))

    #         coeff = '{}{}'.format(i+1, j+1)

    #         # Create histogram
    #         ax = sns.histplot(
    #             data=plot_df.rename(
    #                 columns={'test_case':'Caso de Teste'}
    #             ),
    #             x='b{}'.format(coeff),
    #             hue='Caso de Teste',
    #             hue_order=test_cases,
    #             bins=bin_cfg
    #         )

    #         # Set labels and ticks
    #         plt.ylabel('Contagem de Realizações',fontsize=label_size)
    #         plt.xlabel(r'$\widehat{{b}}_{}$'.format('{'+coeff+'}'),fontsize=label_size)
    #         plt.xticks(fontsize=label_size)
    #         plt.yticks(fontsize=label_size)
    #         plt.setp(
    #             ax.get_legend().get_texts(),
    #             fontsize=legend_size
    #         )

    #         # Plot B_true, if so specified
    #         if B_true is not None:
    #             plt.axvline(
    #                 B_true[i,j],
    #                 color='red',
    #                 linestyle='--'
    #             )

    #         # Save figure, if so specified
    #         if save_dir is not None:
    #             fig.savefig(
    #                 str(
    #                     (save_dir / '{}.png'.format(save_name + '_{}{}'.format(i+1,j+1) if save_name is not None else 'mmse_coeffs'))
    #                 ),
    #                 # bbox_inches='tight'
    #             )

    #     # Calculate statistics, if so specified
    #     if return_stats:
    #         stats_df = plot_df.groupby(
    #             by='test_case',
    #             as_index=False
    #         ).agg(
    #             b11_mean=('b11', 'mean'),
    #             b12_mean=('b12', 'mean'),
    #             b21_mean=('b21', 'mean'),
    #             b22_mean=('b22', 'mean'),
    #             b11_std=('b11', 'std'),
    #             b12_std=('b12', 'std'),
    #             b21_std=('b21', 'std'),
    #             b22_std=('b22', 'std')
    #         )
    #         return stats_df


    @classmethod
    def plot_test_cases_map_coefficient_estimates(
        cls,
        parser,
        test_cases=None,
        B_true=None,
        bin_cfg='auto',
        return_stats=True,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
        # u_range=(-0.3, 0.3),
        # v_range=(-0.3, 0.3),

    ):
        
        plt.rc(
            'legend',
            fontsize=legend_size,
            title_fontsize=legend_size
        )

        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }
    
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():

            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(test_case_results['map']['estimates']),
                            'b11': test_case_results['map']['b11_estimates'],
                            'b12': test_case_results['map']['b12_estimates'],
                            'b21': test_case_results['map']['b21_estimates'],
                            'b22': test_case_results['map']['b22_estimates'],
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )
    
        # Create plot
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(20,15)
        )

        # fig.suptitle(
        #     'Casos de teste: {}'.format(', '.join(test_cases)),
        #     fontsize=25
        # )

        # Plot B_true, if so specified
        if B_true is not None:
            for i,j in np.ndindex(B_true.shape):
                axs[i,j].axvline(
                    B_true[i,j],
                    color='red',
                    linestyle='--'
                )

        # B11
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b11',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[0,0]
        )
        axs[0,0].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[0,0].set_xlabel(r'$\widehat{b}_{11}^{MAP}$',fontsize=label_size)
        axs[0,0].tick_params(
            axis='both',
            labelsize=label_size
        )

        # B12
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b12',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[0,1]
        )
        axs[0,1].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[0,1].set_xlabel(r'$\widehat{b}_{12}^{MAP}$',fontsize=label_size)
        axs[0,1].tick_params(
            axis='both',
            labelsize=label_size
        )

        # B21
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b21',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[1,0]
        )
        axs[1,0].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[1,0].set_xlabel(r'$\widehat{b}_{21}^{MAP}$',fontsize=label_size)
        axs[1,0].tick_params(
            axis='both',
            labelsize=label_size
        )

        # B22
        sns.histplot(
            data=plot_df.rename(columns={'test_case': 'Caso de Teste'}),
            x='b22',
            hue='Caso de Teste',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[1,1]
        )
        axs[1,1].set_ylabel('Contagem de Realizações',fontsize=label_size)
        axs[1,1].set_xlabel(r'$\widehat{b}_{22}^{MAP}$',fontsize=label_size)
        axs[1,1].tick_params(
            axis='both',
            labelsize=label_size
        )

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'map_coeffs'))
                ),
                bbox_inches='tight'
            )

        # Calculate statistics, if so specified
        if return_stats:
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                b11_mean=('b11', 'mean'),
                b12_mean=('b12', 'mean'),
                b21_mean=('b21', 'mean'),
                b22_mean=('b22', 'mean'),
                b11_std=('b11', 'std'),
                b12_std=('b12', 'std'),
                b21_std=('b21', 'std'),
                b22_std=('b22', 'std')
            )
            return stats_df



    @classmethod
    def plot_test_cases_map_errors(
        cls,
        parser,
        test_cases=None,
        bin_cfg='fd',
        return_stats=True,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }
    
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
    
            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(test_case_results['posteriori_grid']['maximums']),
                            'errors': test_case_results['map']['errors']
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )
    
        # Create plot
        fig = plt.figure(figsize=(20,10))
        ax = sns.histplot(
            data=plot_df.rename(
                columns={
                    'test_case': 'Caso de Teste'
                }
            ),
            x='errors',
            hue='Caso de Teste',
            bins=bin_cfg
        )
        
        plt.ylabel('Contagem de Realizações', fontsize=label_size)
        plt.xlabel(r'$\delta_{\bf{B}}({\bf{B}})$', fontsize=label_size)
        plt.xticks(fontsize=label_size)
        plt.yticks(fontsize=label_size)
        plt.setp(
            ax.get_legend().get_texts(),
            fontsize=legend_size
        )
        plt.setp(
            ax.get_legend().get_title(),
            fontsize=legend_size
        )

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'map_errors'))
                ),
                # bbox_inches='tight'
            )

        # Calculate statistics, if so specified
        if return_stats:
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                errors_mean=('errors', 'mean'),
                errors_min=('errors', 'min'),
                errors_max=('errors', 'max'),
                errors_std=('errors', 'std')
            )
            return stats_df
        

    @classmethod
    def plot_test_cases_min_determinants(
        cls,
        parser,
        test_cases=None,
        bin_cfg='fd',
        return_stats=True
    ):

        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }

        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
            # Get mmse objects
            mmse_objects = test_case_results['mmse']['object']

            # Get determinants for samples in each realization
            realizations_determinants = [
                [
                    np.abs(
                        np.linalg.det(B_sample)
                    ) for B_sample in obj.mcmc_results[0]['samples']
                ] for obj in mmse_objects 
            ]

            # Get minimum determinants
            min_determinants = [
                np.min(dets) for dets in realizations_determinants
            ]

            # Get maximum determinants
            max_determinants = [
                np.max(dets) for dets in realizations_determinants
            ]

            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(mmse_objects),
                            'min_abs_det': min_determinants,
                            'max_abs_det': max_determinants,
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )


        # Create plot
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(20,15)
        )
        # Min
        sns.histplot(
            data=plot_df,
            x='min_abs_det',
            hue='test_case',
            bins=bin_cfg,
            ax=ax1
        )
        # Max
        sns.histplot(
            data=plot_df,
            x='max_abs_det',
            hue='test_case',
            bins=bin_cfg,
            ax=ax2
        )

        # Calculate statistics, if so specified
        if return_stats:
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                min_abs_det_min=('min_abs_det', 'min'),
                min_abs_det_max=('min_abs_det', 'max'),
                max_abs_det_min=('max_abs_det', 'min'),
                max_abs_det_max=('max_abs_det', 'max'),
            )
            return stats_df
        

    @classmethod
    def plot_test_cases_determinants_scatter(
        cls,
        parser,
        test_cases=None,
        marker='model'
    ):

        # Model key
        model_key = {
            'i': 'perfect',
            'ii': 'perfect',
            'iii': 'perfect',
            'iv': 'perfect',
            'v': 'slightly_misspecified',
            'vi': 'slightly_misspecified',
            'vii': 'slightly_misspecified',
            'viii': 'slightly_misspecified',
            'ix': 'largely_misspecified',
            'x': 'largely_misspecified',
            'xi': 'largely_misspecified',
            'xii': 'largely_misspecified',
        }

        # Prior key
        prior_key = {
            'i': 'uniform',
            'ii': 'correct_large_variance',
            'iii': 'correct_small_variance',
            'iv': 'incorrect_small_variance',
            'v': 'uniform',
            'vi': 'correct_large_variance',
            'vii': 'correct_small_variance',
            'viii': 'incorrect_small_variance',
            'ix': 'uniform',
            'x': 'correct_large_variance',
            'xi': 'correct_small_variance',
            'xii': 'incorrect_small_variance',
        }
        
        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }

        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
            # Get mmse objects
            mmse_objects = test_case_results['mmse']['object']

            # Get determinants for samples in each realization
            realizations_determinants = [
                [
                    np.abs(
                        np.linalg.det(B_sample)
                    ) for B_sample in obj.mcmc_results[0]['samples']
                ] for obj in mmse_objects 
            ]

            # Get minimum determinants
            min_determinants = [
                np.min(dets) for dets in realizations_determinants
            ]

            # Get maximum determinants
            max_determinants = [
                np.max(dets) for dets in realizations_determinants
            ]

            # Append dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'test_case': [test_case]*len(mmse_objects),
                            'min_abs_det': min_determinants,
                            'max_abs_det': max_determinants,
                        }
                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )

        # Apply model key
        plot_df['model'] = plot_df['test_case'].apply(lambda t: model_key[t])

        # Apply prior key
        plot_df['prior'] = plot_df['test_case'].apply(lambda t: prior_key[t])


        # Create plot
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,7)
        )

        sns.scatterplot(
            data=plot_df,
            x='min_abs_det',
            y='max_abs_det',
            hue='test_case',
            style=marker,
            markers=['s', '>', '<','o'],
            ax=ax
        )


        

        
import os
from pathlib import Path
from pathos.pools import ProcessPool


import pandas as pd
import numpy as np
from scipy import stats as st
from sklearn.feature_selection import mutual_info_regression as MI
from sklearn.mixture import GaussianMixture
from functools import partial

import json
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns


class HypothesisTestsCases:

    def __init__(
        self,
        parser,
        model_type,
        test_cases=['i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii']
    ):
        
        self.parser=parser
        self.model_type=model_type
        self.test_cases=test_cases
        self.model_params = {model_type:{}}
        self.test_results = {}
        self.test_case_mapping = {
            'i': 1,
            'ii': 2,
            'iii': 3,
            'iv': 4,
            'v': 5,
            'vi': 6,
            'vii': 7,
            'viii': 8,
            'ix': 9,
            'x': 10,
            'xi': 11,
            'xii': 12
        }
        self.best_models={}

    def run_shapiro(
        self
    ):
        
        # Iterate in configured tests
        test_results = pd.DataFrame()
        for case in self.test_cases:

            # Get errors
            map_errors = self.parser.parsed_results[case]['map']['errors']
            mmse_errors = self.parser.parsed_results[case]['mmse']['errors']

            # Get sample size
            n_samples = len(map_errors)
            
            # Get shapiro test objects
            map_shapiro = st.shapiro(
                map_errors
            )
            mmse_shapiro = st.shapiro(
                mmse_errors
            )
 
            # Log
            test_results = pd.concat(
                [
                    test_results,
                    pd.DataFrame(
                        index=[0],
                        data={
                            'test_case': [case],
                            'map_error_shapiro_statistic': [map_shapiro.statistic],
                            'map_error_shapiro_pvalue': [map_shapiro.pvalue],
                            'mmse_error_shapiro_statistic': [mmse_shapiro.statistic],
                            'mmse_error_shapiro_pvalue': [mmse_shapiro.pvalue]
                        }

                    )
                ],
                axis=0
            ).reset_index(
                drop=True
            )
        
        self.shapiro_results = test_results

    def error_probplot(
        self,
        estimate_type,
        fontsize=20
    ):

        plt.rc(
            'legend',
            fontsize=fontsize,
            title_fontsize=fontsize
        )

        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'

        plt.rc('axes', labelsize=fontsize)
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize) 
        
        fig, axs = plt.subplots(
            nrows=6, ncols=2,
            figsize=(20,60)
        )

        # Iterate in configured tests
        for case in self.test_cases:
            
            case_number = self.test_case_mapping[case]

            shapiro_pvalue = self.shapiro_results.at[case_number-1, '{}_error_shapiro_pvalue'.format(estimate_type)]

            axis_row = (case_number-1)//2
            axis_col = (case_number-1)%2


            st.probplot(
                self.parser.parsed_results[case][estimate_type]['errors'],
                dist='norm',
                plot=axs[axis_row, axis_col]
            )

            axs[axis_row, axis_col].set_title(
                'Caso de teste {}. p-valor Shapiro-Wilk: {}'.format(
                    case,
                    shapiro_pvalue
                ),
                fontsize=fontsize
            )

            axs[axis_row, axis_col].set_xlabel(
                'Quantil teórico - normal padrão',
                fontsize=fontsize
            )

            axs[axis_row, axis_col].set_ylabel(
                'Amostras observadas',
                fontsize=fontsize
            )

            for spine in ['bottom', 'top', 'left', 'right']:
                axs[axis_row, axis_col].spines[spine].set_color('black')


    def fit_gamma(
        self,
        estimate_type,
    ):
        model_params = {}
        for case in self.test_cases:
            # Fit gamma params
            params = st.gamma.fit(
                data=self.parser.parsed_results[case][estimate_type]['errors'],
                floc=0
            )
            model_params[case] = {
                'alpha': params[0],
                'beta': 1/params[-1]
            }

            # Plot pdf vs histogram
            plot_x = np.linspace(
                min(self.parser.parsed_results[case][estimate_type]['errors']),
                max(self.parser.parsed_results[case][estimate_type]['errors']),
                100
            )

            fig, ax = plt.subplots(
                1,1,
                figsize=(15,5)
            )
            ax.set_title(
                'Estimação {}, Caso de teste {}: alpha={}, beta={}'.format(
                    estimate_type,
                    case,
                    params[0],
                    1/params[-1]
                ),
                fontsize=15
            )
            ax.plot(
                plot_x,
                st.gamma.pdf(
                    plot_x, 
                    a=params[0], 
                    loc=0, 
                    scale=params[-1],
                ),
                color='cornflowerblue'
            )
            ax.hist(
                self.parser.parsed_results[case][estimate_type]['errors'],
                density=True,
                color='cornflowerblue',
                alpha=0.7
            )

        # Save to object attribute
        self.model_params[self.model_type][estimate_type]=model_params


    def fit_gmm(
        self,
        estimate_type,
        n_components=None,
        seeds=[1],
        save_dir=None,
        save_name=None,
    ):
        
        def __gmm_likelihood(model, x):

            model_likelihoods = np.array([0]*len(x))
            for c in range(model.n_components):
                # Get weighted pdf values for gaussian component at data points
                component_likelihoods = np.array([
                    model.weights_[c]*y for y in st.norm.pdf(
                        x,
                        loc=model.means_[c][0],
                        scale=np.sqrt(model.covariances_[c][0])
                    )
                ])

                # Elementwise add to get total pdf value at each data point
                model_likelihoods = np.add(
                    model_likelihoods,
                    component_likelihoods
                )

            return np.sum([
                np.log(y) for y in model_likelihoods
            ])
        
        def __fit_gmm(
            n_components,
            errors,
            seed
        ):
            gmm  = GaussianMixture(
                n_components=n_components,
                random_state=seed
            ).fit(
                errors
            )

            return gmm
            

        fig, axs = plt.subplots(
            nrows=6,ncols=2,
            figsize=(20,30)
        )

        model_params = {}
        for case in self.test_cases:
            
            case_number = self.test_case_mapping[case]

            # Fixate parameters in fit fn
            fit_fn = partial(
                __fit_gmm,
                self.best_models[estimate_type][case] if n_components is None else n_components,
                np.array(self.parser.parsed_results[case][estimate_type]['errors']).reshape(-1,1)
            )

            # Fit models with different seeds
            models = [
                fit_fn(seed) for seed in seeds
            ]
            # with ProcessPool(n_workers) as p:
            #     models = p.map(
            #         fit_fn,
            #         seeds
            #     )

            # Get model likelihoods
            models_likelihoods = [
                __gmm_likelihood(
                    model=model, 
                    x=self.parser.parsed_results[case][estimate_type]['errors']
                ) for model in models
            ]

            # Get best model
            gmm = models[
                np.argmax(models_likelihoods)
            ]

            # Save likelihood method to gmm object
            gmm.likelihood = partial(
                __gmm_likelihood,
                gmm
            )

            model_params[case]=gmm

            # Plot pdf vs histogram
            plot_x = np.linspace(
                min(self.parser.parsed_results[case][estimate_type]['errors']),
                max(self.parser.parsed_results[case][estimate_type]['errors']),
                100
            )
            
            axis_row = (case_number-1)//2
            axis_col = (case_number-1)%2

            axs[axis_row, axis_col].set_title(
                'Caso de teste {}: {} gaussianas'.format(
                    case,
                    gmm.n_components
                ),
                fontsize=15
            )

            axs[axis_row, axis_col].hist(
                self.parser.parsed_results[case][estimate_type]['errors'],
                density=True,
                color='cornflowerblue',
                alpha=0.7
            )

            plot_pdf = [0]*len(plot_x)
            for c in range(gmm.n_components):

                component_pdf = [
                    gmm.weights_[c]*y for y in st.norm.pdf(
                        plot_x,
                        loc=gmm.means_[c][0],
                        scale=np.sqrt(gmm.covariances_[c][0])
                    )
                ]

                plot_pdf = np.add(plot_pdf, component_pdf)

                axs[axis_row, axis_col].plot(
                    plot_x,
                    component_pdf,
                    linestyle='--',
                    color='cornflowerblue'
                )

            axs[axis_row, axis_col].plot(
                plot_x,
                plot_pdf,
                color='cornflowerblue'
            )

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'gmm_fit_{}'.format(estimate_type)))
                ),
                bbox_inches='tight'
            )
            

        # Save to object attribute
        self.model_params[self.model_type][estimate_type]=model_params
    
    def bic_analysis(
        self,
        n_components,
        estimate_type,
        save_dir=None,
        save_name=None,
    ):
        bic_df = pd.DataFrame()
        best_models = {}
        # Iterate in test cases and component numbers and fit GMMs
        for n in n_components:
            for case in self.test_cases:
                # Fit GMM
                gmm = GaussianMixture(
                    n_components=n
                ).fit(
                    np.array(self.parser.parsed_results[case][estimate_type]['errors']).reshape(-1,1)
                )

                # Log results
                bic_df = pd.concat(
                    [
                        bic_df,
                        pd.DataFrame(
                            index=[0],
                            data={
                                'estimate_type': [estimate_type],
                                'n_gaussians': [n],
                                'test_case': [case],
                                'bic': gmm.bic(np.array(self.parser.parsed_results[case][estimate_type]['errors']).reshape(-1,1))
                            }
                        )
                    ],
                    axis=0
                ).reset_index(
                    drop=True
                )

        colors = [
            'green',
            'darkorange',
            'red'
        ]

        markers = [
            's',
            '>',
            '<',
            'o'
        ]

        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,7)
        )

        for case in self.test_cases:
            # Filter test case
            case_df = bic_df[
                bic_df.test_case==case
            ]

            # Get test case in number
            case_number = self.test_case_mapping[case]

            # Get x for plot
            case_gaussians = case_df['n_gaussians'].values

            # Get y for plot
            case_bic = case_df['bic'].values

            # Get best model for test case and save
            best_model = case_df.loc[case_df.bic.idxmin()]['n_gaussians']
            best_models[case]=int(best_model)
            
            #

            ax.plot(
                case_gaussians,
                case_bic,
                color=colors[(case_number-1)//4],
                marker=markers[case_number%4],
                linestyle='--',
                alpha=0.6,
                label=case
            )

        ax.set_xlabel(
            'Número de Gaussianas',
            fontsize=15
        )

        ax.set_ylabel(
            'BIC',
            fontsize=15
        )

        ax.legend(title='Caso de Teste')

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'bic_analysis_{}'.format(estimate_type)))
                ),
                bbox_inches='tight'
            )

        self.best_models[estimate_type] = best_models



    def run_compare_cases(
        self,
        estimate_type,
        tests,
        means_test=None
    ):
        
        def __get_lrt_statistic_gamma(
            ref_case,
            alt_case,
            estimate_type
        ):
            # Get likelihood of alt case errors with alt case params
            alt_case_log_likelihood = np.sum([
                np.log(st.gamma.pdf(
                    e, 
                    a=self.model_params[self.model_type][estimate_type][alt_case]['alpha'], 
                    loc=0, 
                    scale=1/self.model_params[self.model_type][estimate_type][alt_case]['beta']
                )) for e in self.parser.parsed_results[alt_case][estimate_type]['errors']
            ])

            # Get likelihood of alt case errors with ref case params
            ref_case_log_likelihood = np.sum([
                np.log(st.gamma.pdf(
                    e,
                    a=self.model_params[self.model_type][estimate_type][ref_case]['alpha'], 
                    loc=0, 
                    scale=1/self.model_params[self.model_type][estimate_type][ref_case]['beta']
                )) for e in self.parser.parsed_results[alt_case][estimate_type]['errors']
            ])

            # Get test statistic
            test_statistic = np.exp(ref_case_log_likelihood - alt_case_log_likelihood)

            return pd.DataFrame(
                index=[0],
                data={
                    'ref_case': [ref_case],
                    'alt_case': [alt_case],
                    'ref_case_alpha': [self.model_params[self.model_type][estimate_type][ref_case]['alpha']],
                    'ref_case_beta': [self.model_params[self.model_type][estimate_type][ref_case]['beta']],
                    'alt_case_alpha': [self.model_params[self.model_type][estimate_type][alt_case]['alpha']],
                    'alt_case_beta': [self.model_params[self.model_type][estimate_type][alt_case]['beta']],
                    'ref_case_log_likelihood': [ref_case_log_likelihood],
                    'alt_case_log_likelihood': [alt_case_log_likelihood],
                    'test_statistic': [test_statistic]
                }
            )
        
        def __get_lrt_statistic_gmm(
            ref_case,
            alt_case,
            estimate_type
        ):
            # Get models for alt case and ref case  
            alt_gmm = self.model_params[self.model_type][estimate_type][alt_case]
            ref_gmm = self.model_params[self.model_type][estimate_type][ref_case]
            
            # Get likelihood of alt case errors with alt case params
            alt_case_log_likelihood = alt_gmm.likelihood(self.parser.parsed_results[alt_case][estimate_type]['errors'])

            # Get likelihood of alt case errors with ref case params
            ref_case_log_likelihood = ref_gmm.likelihood(self.parser.parsed_results[alt_case][estimate_type]['errors'])

            # Get test statistic
            test_statistic = np.exp(ref_case_log_likelihood - alt_case_log_likelihood)

            return pd.DataFrame(
                index=[0],
                data={
                    'ref_case': [ref_case],
                    'alt_case': [alt_case],
                    'ref_case_log_likelihood': [ref_case_log_likelihood],
                    'alt_case_log_likelihood': [alt_case_log_likelihood],
                    'lrt_statistic': [test_statistic]
                }
            )
        
        # Iterate in configured tests
        test_results = pd.DataFrame()
        for t in tests:
            # Parse test configs
            ref_case=t['ref_case']
            alt_cases=t['alt_cases']
            # Run LRT      
            for alt_case in alt_cases:
                if self.model_type=='gamma':
                    # Get results using gamma pdf model
                    case_results = __get_lrt_statistic_gamma(
                        ref_case=ref_case,
                        alt_case=alt_case,
                        estimate_type=estimate_type
                    )
                elif self.model_type=='gmm':
                    case_results = __get_lrt_statistic_gmm(
                        ref_case=ref_case,
                        alt_case=alt_case,
                        estimate_type=estimate_type
                    )

                # Run means test
                if means_test is not None:
                    if means_test=='independent_t':
                        means_results = st.ttest_ind(
                            a=self.parser.parsed_results[ref_case][estimate_type]['errors'],
                            b=self.parser.parsed_results[alt_case][estimate_type]['errors']
                        )


                        # Log means test results
                        case_results['ref_case_error_mean'] = np.mean(self.parser.parsed_results[ref_case][estimate_type]['errors'])
                        case_results['alt_case_error_mean'] = np.mean(self.parser.parsed_results[alt_case][estimate_type]['errors'])
                        case_results['t_test_statistic'] = means_results.statistic
                        case_results['t_test_pvalue'] = means_results.pvalue
                    elif means_test=='paired_t':
                        # Run means test
                        differences = np.subtract(
                            self.parser.parsed_results[ref_case][estimate_type]['errors'],
                            self.parser.parsed_results[alt_case][estimate_type]['errors']
                        )
                        degrees_of_freedom = len(self.parser.parsed_results[ref_case][estimate_type]['errors']) - 1
                        shapiro = st.shapiro(
                            differences
                        )
                        # means_results = st.ttest_rel(
                        #     self.parser.parsed_results[case]['map']['errors'],
                        #     self.parser.parsed_results[case]['mmse']['errors']
                        # )

                        # Log means test results
                        case_results['ref_case_error_mean'] = np.mean(self.parser.parsed_results[ref_case][estimate_type]['errors'])
                        case_results['alt_case_error_mean'] = np.mean(self.parser.parsed_results[alt_case][estimate_type]['errors'])
                        case_results['difference_mean'] = np.mean(differences)
                        case_results['difference_std'] = np.std(differences)
                        case_results['shapiro_statistic'] = shapiro.statistic
                        case_results['shapiro_pvalue'] = shapiro.pvalue
                        case_results['t_test_statistic'] = np.sqrt(len(self.parser.parsed_results[ref_case][estimate_type]['errors']))*case_results['difference_mean']/case_results['difference_std']
                        case_results['t_test_pvalue'] = case_results['t_test_statistic'].apply(
                            lambda s: (st.t.cdf(-1*s, degrees_of_freedom) + (1-st.t.cdf(s, degrees_of_freedom))) if s>=0 else (st.t.cdf(s, degrees_of_freedom) + (1-st.t.cdf(-1*s, degrees_of_freedom)))
                        )
                    
                    if means_test=='approximate_z_score':
                        # Save errors
                        ref_errors = self.parser.parsed_results[ref_case][estimate_type]['errors']
                        alt_errors = self.parser.parsed_results[alt_case][estimate_type]['errors']

                        # Get sample size
                        n_samples = len(ref_errors)

                        # Get means and standard deviations
                        ref_mean, ref_std = np.mean(ref_errors), np.std(ref_errors)
                        alt_mean, alt_std = np.mean(alt_errors), np.std(alt_errors)

                        # Get absolute difference in means, wrt to approximate mean uncertainty
                        abs_mean_difference = np.abs(alt_mean - ref_mean)
                        sum_std = ref_std + alt_std
                        mean_diff_wrt_std = np.sqrt(n_samples)*abs_mean_difference/sum_std



                        # Save to dataframes
                        case_results['ref_case_error_mean'] = ref_mean
                        case_results['alt_case_error_mean'] = alt_mean
                        case_results['ref_case_error_std'] = ref_std
                        case_results['alt_case_error_std'] = alt_std
                        case_results['abs_mean_difference'] = abs_mean_difference
                        case_results['sum_std'] = sum_std
                        case_results['mean_diff_wrt_std'] = mean_diff_wrt_std

                    

                # Log
                test_results = pd.concat(
                    [
                        test_results,
                        case_results
                    ],
                    axis=0
                ).reset_index(
                    drop=True
                )

        return test_results
    

    def run_compare_map_mmse(
        self,
        means_test=None,
        save_dir=None,
        save_name=None
    ):
        
        def __get_lrt_statistic_gmm(
            case
        ):
            # Get models for map and mmse 
            map_gmm = self.model_params[self.model_type]['map'][case]
            mmse_gmm = self.model_params[self.model_type]['mmse'][case]
            
            # Get likelihood of alt case errors with alt case params
            mmse_log_likelihood = mmse_gmm.likelihood(self.parser.parsed_results[case]['mmse']['errors'])

            # Get likelihood of alt case errors with ref case params
            map_log_likelihood = map_gmm.likelihood(self.parser.parsed_results[case]['mmse']['errors'])

            # Get test statistic
            test_statistic = np.exp(map_log_likelihood - mmse_log_likelihood)

            return pd.DataFrame(
                index=[0],
                data={
                    'case': [case],
                    'map_log_likelihood': [map_log_likelihood],
                    'mmse_log_likelihood': [mmse_log_likelihood],
                    'lrt_statistic': [test_statistic]
                }
            )
        
        fig, axs = plt.subplots(
            nrows=6, ncols=2,
            figsize=(20,30)
        )

        # Iterate in configured tests
        test_results = pd.DataFrame()
        for case in self.test_cases:
            if self.model_type=='gamma':
                pass
            elif self.model_type=='gmm':
                case_results = __get_lrt_statistic_gmm(
                    case=case
                )

            if means_test is not None:
                if means_test=='paired_t_test':
                    # Run means test
                    differences = np.subtract(
                        self.parser.parsed_results[case]['map']['errors'],
                        self.parser.parsed_results[case]['mmse']['errors']
                    )
                    degrees_of_freedom = len(self.parser.parsed_results[case]['map']['errors']) - 1
                    shapiro = st.shapiro(
                        differences
                    )
                    # means_results = st.ttest_rel(
                    #     self.parser.parsed_results[case]['map']['errors'],
                    #     self.parser.parsed_results[case]['mmse']['errors']
                    # )

                    # Log means test results
                    case_results['map_error_mean'] = np.mean(self.parser.parsed_results[case]['map']['errors'])
                    case_results['mmse_error_mean'] = np.mean(self.parser.parsed_results[case]['mmse']['errors'])
                    case_results['difference_mean'] = np.mean(differences)
                    case_results['difference_std'] = np.std(differences)
                    case_results['shapiro_statistic'] = shapiro.statistic
                    case_results['shapiro_pvalue'] = shapiro.pvalue
                    case_results['t_test_statistic'] = np.sqrt(len(self.parser.parsed_results[case]['map']['errors']))*case_results['difference_mean']/case_results['difference_std']
                    case_results['t_test_pvalue'] = case_results['t_test_statistic'].apply(
                        lambda s: (st.t.cdf(-1*s, degrees_of_freedom) + (1-st.t.cdf(s, degrees_of_freedom))) if s>=0 else (st.t.cdf(s, degrees_of_freedom) + (1-st.t.cdf(-1*s, degrees_of_freedom)))
                    )
                elif means_test=='approximate_z_score':
                    # Save errors
                    map_errors = self.parser.parsed_results[case]['map']['errors']
                    mmse_errors = self.parser.parsed_results[case]['mmse']['errors']

                    # Get sample size
                    n_samples = len(map_errors)

                    # Get means and standard deviations
                    map_mean, map_std = np.mean(map_errors), np.std(map_errors)
                    mmse_mean, mmse_std = np.mean(mmse_errors), np.std(mmse_errors)

                    # Get absolute difference in means, wrt to approximate mean uncertainty
                    abs_mean_difference = np.abs(mmse_mean - map_mean)
                    sum_std = mmse_std + map_std
                    mean_diff_wrt_std = np.sqrt(n_samples)*abs_mean_difference/sum_std



                    # Save to dataframes
                    case_results['map_error_mean'] = map_mean
                    case_results['mmse_error_mean'] = mmse_mean
                    case_results['map_error_std'] = map_std
                    case_results['mmse_error_std'] = mmse_std
                    case_results['abs_mean_difference'] = abs_mean_difference
                    case_results['sum_std'] = sum_std
                    case_results['mean_diff_wrt_std'] = mean_diff_wrt_std

                
            # Log
            test_results = pd.concat(
                [
                    test_results,
                    case_results
                ],
                axis=0
            ).reset_index(
                drop=True
            )

            case_number = self.test_case_mapping[case]

            axis_row = (case_number-1)//2
            axis_col = (case_number-1)%2

            axs[axis_row, axis_col].set_title(
                'Caso de teste {}'.format(
                    case
                ),
                fontsize=15
            )

            axs[axis_row, axis_col].hist(
                np.subtract(
                    self.parser.parsed_results[case]['map']['errors'],
                    self.parser.parsed_results[case]['mmse']['errors']
                ),
                density=True,
                color='cornflowerblue',
                alpha=0.7
            )
        
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'compare_map_mmse'))
                ),
                bbox_inches='tight'
            )

        return test_results
    
    def plot_errors_boxplot_map_mmse(
        self,
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
        legend_loc='upper left'
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
        parsed_results = {
            k: v for k, v in self.parser.parsed_results.items() if k in self.test_cases
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
                                'estimate_type': [estimate_type.upper()]*len(test_case_results[estimate_type]['estimates']),
                                'test_case': [test_case]*len(test_case_results[estimate_type]['estimates']),
                                'b11': test_case_results[estimate_type]['b11_estimates'],
                                'b12': test_case_results[estimate_type]['b12_estimates'],
                                'b21': test_case_results[estimate_type]['b21_estimates'],
                                'b22': test_case_results[estimate_type]['b22_estimates'],
                                'error': test_case_results[estimate_type]['errors']
                            }
                        )
                    ],
                    axis=0
                ).reset_index(
                    drop=True
                )


        # Calculate stats_df
        stats_df = plot_df.groupby(
            by=['estimate_type','test_case'],
            as_index=False
        ).agg(
            b11_mean=('b11', 'mean'),
            b12_mean=('b12', 'mean'),
            b21_mean=('b21', 'mean'),
            b22_mean=('b22', 'mean'),
            error_mean=('error', 'mean'),
            b11_std=('b11', 'std'),
            b12_std=('b12', 'std'),
            b21_std=('b21', 'std'),
            b22_std=('b22', 'std'),
            error_std=('error', 'std')
        )

        # Create plot
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,7)
        )

        # Plot horizontal line at 0
        ax.axhline(
            0,
            color='lightgray',
            linestyle='--'
        )

        sns.boxplot(
            data=plot_df.rename(
                columns={'estimate_type': 'Tipo de Estimativa'}
            ),
            x='test_case',
            y='error',
            hue='Tipo de Estimativa',
            # color=box_color,
            # linecolor=box_linecolor,
            # linewidth=box_linewidth,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax
        )

        ax.legend(
            fontsize=legend_size,
            loc=legend_loc
        )
        
        ax.set_ylabel(
            r'$\delta_{\bf{B}}(\widehat{\bf{B}})$'
        )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_color('black')
        
        ax.set_xlabel(
            'Caso de Teste'
        )
        
        if yranges is not None:
            ax.set_ylim(
                yranges[0],
                yranges[-1],
            )
        
        # Plot vertical lines
        [
            ax.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax.get_xticks()
        ]


            
        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'errors_boxplots'))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df

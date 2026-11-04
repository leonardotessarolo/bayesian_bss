import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns

class VisualizationUtilities:
    @staticmethod
    def plot_test_cases_contours(
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
        
    @staticmethod
    def plot_recovery_errors(
        parser,
        error_type,
        estimate_type,
        test_cases=None,
        return_stats=True,
        whis_lims=(5,95),
        box_color='gray',
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
        for test_case, test_case_results in parsed_results.items():

            # Create estimate dataframe
            plot_df = pd.concat(
                [
                    plot_df,
                    pd.DataFrame(
                        data={
                            'estimate_type': [estimate_type]*parser.effective_realizations,
                            'test_case': [test_case]*parser.effective_realizations,
                            error_type: test_case_results[estimate_type][error_type]
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
            error_mean=(error_type, 'mean'),
            error_std=(error_type, 'std')
        )

        # Create plot
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,5)
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
            y=error_type,
            color=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax
        )
        if error_type=='s_errors':
            ax.set_ylabel(
                r'$\delta_{\bf{S}}(\widehat{\bf{S}})$'
            )
        elif error_type=='B_errors':
            ax.set_ylabel(
                r'$\delta_{\bf{B}}(\widehat{\bf{B}})$'
            )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_color('black')
        
        ax.set_xlabel(
            'Caso de Teste'
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
                    (save_dir / '{}.png'.format(save_name if save_name is not None else '{}_{}_boxplots'.format(estimate_type, error_type)))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df
    
    @staticmethod
    def mcmc_diagnostics_analysis(
        parser,
        test_cases=None,
        return_stats=True,
        whis_lims=(5,95),
        box_color='gray',
        log_scale=False,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
    ):
        def __get_diagnostic(
            diag,
            test_case,
            test_case_results
        ):
            diag_columns = [
                c for c in test_case_results['mmse']['diagnostics'][0].columns if diag in c
            ]
            test_case_diagnostics = pd.DataFrame([
                {
                    c: df.iloc[-1,:][c] for c in diag_columns
                } for df in test_case_results['mmse']['diagnostics']
            ]).melt(
                value_vars=diag_columns,
                var_name='coefficient',
                value_name=diag
            )
            test_case_diagnostics['test_case'] = test_case
            test_case_diagnostics['coefficient'] = test_case_diagnostics['coefficient'].str.slice(-4)
            test_case_diagnostics = test_case_diagnostics[[
                'test_case', 'coefficient', diag
            ]]

            return test_case_diagnostics
            
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
        
        # Loop over test cases and extract last R-hat/ESS value
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
            # Get diagnostics
            diagnostics = [
                __get_diagnostic(
                    diag=d,
                    test_case=test_case,
                    test_case_results=test_case_results
                ) for d in ['r_hat', 'ess']
            ]

            # Merge diagnostics dataframes
            diagnostics_df = diagnostics[0].merge(
                diagnostics[-1],
                on=['test_case', 'coefficient']
            )
            
            plot_df = pd.concat(
                [
                    plot_df,
                    diagnostics_df
                ],
                axis=0
            )

            
        # Calculate stats_df
        stats_df = plot_df.groupby(
            by=['test_case','coefficient'],
            as_index=False
        ).agg(
            r_hat_mean=('r_hat', 'mean'),
            r_hat_std=('r_hat', 'std'),
            r_hat_min=('r_hat', 'min'),
            r_hat_max=('r_hat', 'max'),
            r_hat_median=('r_hat', 'median'),
            ess_mean=('ess', 'mean'),
            ess_std=('ess', 'std'),
            ess_min=('ess', 'min'),
            ess_max=('ess', 'max'),
            ess_median=('ess', 'median'),
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(20,15)
        )
        if log_scale:
            ax1.set_yscale('log')
            ax2.set_yscale('log')

        # Plot horizontal line at 1
        ax1.axhline(
            1,
            color='lightgray',
            linestyle='--'
        )

        # Plot horizontal line at r-hat thresh
        ax1.axhline(
            parser.execution_config['mcmc']['R_hat_thresh'],
            color='lightgray',
            linestyle='--'
        )

        sns.boxplot(
            data=plot_df,
            x='test_case',
            y='r_hat',
            hue='coefficient',
            color=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax1
        )
        ax1.set_ylabel(
            r'$\widehat{R}$'
        )

        sns.boxplot(
            data=plot_df,
            x='test_case',
            y='ess',
            hue='coefficient',
            color=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax2
        )
        ax2.set_ylabel(
            r'$ESS$'
        )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax1.spines[spine].set_color('black')
            ax2.spines[spine].set_color('black')
        
        ax1.set_xlabel(
            'Caso de Teste'
        )
        ax2.set_xlabel(
            'Caso de Teste'
        )
        
        
        # Plot vertical lines
        [
            ax1.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax1.get_xticks()
        ]
        [
            ax2.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax2.get_xticks()
        ]

        
    
        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'mcmc_diagnostics_boxplots'))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
                return stats_df
    
    @staticmethod
    def mcmc_trace_plots(
        parser,
        B_true,
        test_cases=None,
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

        parallel_chains = parser.execution_config['mcmc']['parallel_chains']
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }

        for test_case, test_case_results in parser.parsed_results.items():
            # Plot chain evolutions
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,20)
            )
            fig.suptitle(
                'Caso de teste: {}'.format(test_case),
                fontsize=label_size
            )
            samples_realizations = test_case_results['mmse']['samples']
            for i,j in np.ndindex(B_true.shape):
                for samples in samples_realizations:
                    for c in range(parallel_chains):
                        axs[i,j].plot(
                            samples[c][:,i,j],
                            # label=c
                            linewidth=.5,
                            # color='cornflowerblue'
                        )
                # axs[i,j].legend(
                #     fontsize=15
                # )
                axs[i,j].set_xlabel(
                    'iteração',
                    fontsize=15
                )
                axs[i,j].axhline(
                    B_true[i,j],
                    linestyle='--',
                    color='gray'
                )
                axs[i,j].set_ylabel(
                    'b{}{}'.format(
                        i+1,
                        j+1
                    ),
                    fontsize=15
                )

            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'trace_plots_{}'.format(test_case)))
                    ),
                    bbox_inches='tight'
                )
    
    @staticmethod
    def grad_asc_trace_plots(
        parser,
        B_true,
        test_cases=None,
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
        
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }

        for test_case, test_case_results in parser.parsed_results.items():
            # Plot chain evolutions
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,20)
            )
            fig.suptitle(
                'Caso de teste: {}'.format(test_case),
                fontsize=label_size
            )
            B_estimates = [
                np.stack(logs.B.values) for logs in test_case_results['map']['logs']
            ]
            for i,j in np.ndindex(B_true.shape):
                for B_est in B_estimates:
                        axs[i,j].plot(
                            (np.array(B_est[:,i,j]) - B_true[i,j])/np.abs(B_true[i,j]),
                            # label=c
                            linewidth=.5,
                            color='cornflowerblue'
                        )
                # axs[i,j].legend(
                #     fontsize=15
                # )
                axs[i,j].set_xlabel(
                    'iteração',
                    fontsize=15
                )
                # axs[i,j].axhline(
                #     B_true[i,j],
                #     linestyle='--',
                #     color='gray'
                # )
                axs[i,j].axhline(
                    0,
                    linestyle='--',
                    color='gray'
                )
                axs[i,j].set_ylabel(
                    'b{}{}'.format(
                        i+1,
                        j+1
                    ),
                    fontsize=15
                )
            
            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'coefficients_evolutions_{}'.format(test_case)))
                    ),
                    bbox_inches='tight'
                )

    @staticmethod
    def gradient_ascent_posteriori_evolutions(
        parser,
        test_cases=None,
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
        
        
        for test_case, test_case_results in parsed_results.items():
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(20,7)
            )
            fig.suptitle(
                test_case,
                fontsize=label_size
            )
            ax.set_xlabel(
                'iteration',
                fontsize=label_size
            )
            ax.set_ylabel(
                'normalized log-posterior',
                fontsize=label_size
            )
            for logs in test_case_results['map']['logs']:
                ax.plot(
                    logs.iteration.values,
                    logs.log_posterior.values,
                    linewidth=.5,
                    color='cornflowerblue'
                )

            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'posteriori_evolutions_{}'.format(test_case)))
                    ),
                    bbox_inches='tight'
                )

    @staticmethod
    def plot_recovery_errors_map_mmse(
        parser,
        error_type,
        test_cases=None,
        return_stats=True,
        whis_lims=(5,95),
        palette=None,
        estimate_types=('map', 'mmse'),
        legend_location=None,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
    ):
        """
            Boxplots dos erros de recuperação (error_type = 'B_errors' ou 's_errors') por caso de teste,
            com as caixas de MAP e MMSE lado a lado (seaborn `hue`).
        """
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
        # Rótulos exibidos na legenda
        estimate_labels = {'map': 'MAP', 'mmse': 'MMSE'}
        hue_order = [estimate_labels[e] for e in estimate_types]
        # Filter out desired test cases, if so specified
        parsed_results = parser.parsed_results
        if test_cases is not None:
            parsed_results = {
                k: v for k, v in parser.parsed_results.items() if k in test_cases
            }
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
            for estimate_type in estimate_types:
                errors = np.asarray(test_case_results[estimate_type][error_type])
                plot_df = pd.concat(
                    [
                        plot_df,
                        pd.DataFrame(
                            data={
                                'estimate_type': [estimate_labels[estimate_type]]*len(errors),
                                'test_case': [test_case]*len(errors),
                                error_type: errors
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
            error_mean=(error_type, 'mean'),
            error_std=(error_type, 'std')
        )
        # Create plot
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,5)
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
            y=error_type,
            hue='Tipo de Estimativa',
            hue_order=hue_order,
            order=list(parsed_results.keys()),
            palette=palette,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax
        )
        if error_type=='s_errors':
            ax.set_ylabel(
                r'$\delta_{\bf{S}}(\widehat{\bf{S}})$'
            )
        elif error_type=='B_errors':
            ax.set_ylabel(
                r'$\delta_{\bf{B}}(\widehat{\bf{B}})$'
            )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_color('black')
        ax.set_xlabel(
            'Caso de Teste'
        )
        # Legend
        if legend_location is None:
            # Por padrão, legenda acima do gráfico (não sobrepõe as caixas)
            sns.move_legend(
                ax,
                'lower center',
                bbox_to_anchor=(0.5, 1.0),
                ncol=len(estimate_types),
                frameon=False
            )
        else:
            sns.move_legend(ax, legend_location)
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
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'map_mmse_{}_boxplots'.format(error_type)))
                ),
                bbox_inches='tight'
            )
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df



    @staticmethod
    def posterior_properties_analysis(
        parser,
        true_B,
        test_cases=None,
        return_stats=True,
        whis_lims=(5,95),
        box_color='gray',
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
    ):
        def __get_properties(
            test_case,
            test_case_results,
            true_B
        ):
            mmse = test_case_results['mmse']

            stacked = {
                var: np.stack(mmse[var]) for var in [
                    'B_estimates',
                    'B_var',
                    'B_skew',
                    'B_kurt'
                ]
            }

            frames = []
            for i, j in np.ndindex(true_B.shape):
                frames.append(pd.DataFrame({
                    'test_case': test_case,
                    'coefficient': f'b{i+1}{j+1}',
                    'B_estimates': (stacked['B_estimates'][:, i, j] - true_B[i, j])/np.abs(true_B[i, j]),
                    'B_var': stacked['B_var'][:, i, j],
                    'B_skew': stacked['B_skew'][:, i, j],
                    'B_kurt': stacked['B_kurt'][:, i, j],
                }))

            return pd.concat(frames, ignore_index=True)
        
            
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
        
        # Loop over test cases and extract last R-hat/ESS value
        plot_df = pd.DataFrame()
        for test_case, test_case_results in parsed_results.items():
            # Get properties
            properties_df = __get_properties(
                test_case=test_case,
                test_case_results=test_case_results,
                true_B=true_B
            )

            plot_df = pd.concat(
                [
                    plot_df,
                    properties_df
                ],
                axis=0
            ).reset_index(
                drop=True
            )

        # Calculate stats_df
        stats_df = plot_df.groupby(
            by=['test_case','coefficient'],
            as_index=False
        ).agg(
            b_est_mean=('B_estimates', 'mean'),
            b_est_std=('B_estimates', 'std'),
            b_var_mean=('B_var', 'mean'),
            b_var_std=('B_var', 'std'),
            b_skew_mean=('B_skew', 'mean'),
            b_skew_std=('B_skew', 'std'),
            b_kurt_mean=('B_kurt', 'mean'),
            b_kurt_std=('B_kurt', 'std')
        )

        # Create plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=4, ncols=1,
            figsize=(20,40)
        )

        # Plot horizontal line at zero for odd moments
        ax1.axhline(
            0,
            color='lightgray',
            linestyle='--'
        )
        ax3.axhline(
            0,
            color='lightgray',
            linestyle='--'
        )

        sns.boxplot(
            data=plot_df.rename(
                columns={'B_estimates': 'mean'}
            ),
            x='test_case',
            y='mean',
            hue='coefficient',
            palette=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax1
        )

        sns.boxplot(
            data=plot_df.rename(
                columns={'B_var': 'variance'}
            ),
            x='test_case',
            y='variance',
            hue='coefficient',
            palette=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax2
        )

        sns.boxplot(
            data=plot_df.rename(
                columns={'B_skew': 'skew'}
            ),
            x='test_case',
            y='skew',
            hue='coefficient',
            palette=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax3
        )

        sns.boxplot(
            data=plot_df.rename(
                columns={'B_kurt': 'kurtosis'}
            ),
            x='test_case',
            y='kurtosis',
            hue='coefficient',
            palette=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax4
        )

        
        for spine in ['bottom', 'top', 'left', 'right']:
            ax1.spines[spine].set_color('black')
            ax2.spines[spine].set_color('black')
            ax3.spines[spine].set_color('black')
            ax4.spines[spine].set_color('black')
        
        ax1.set_xlabel(
            'Caso de Teste'
        )
        ax2.set_xlabel(
            'Caso de Teste'
        )
        ax3.set_xlabel(
            'Caso de Teste'
        )
        ax4.set_xlabel(
            'Caso de Teste'
        )
        
        
        # Plot vertical lines
        [
            ax1.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax1.get_xticks()
        ]
        [
            ax2.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax2.get_xticks()
        ]
        [
            ax3.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax3.get_xticks()
        ]
        [
            ax4.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax4.get_xticks()
        ]

        

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'posterior_properties'))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
                return stats_df
        

    def IS_trace_plots(
        parser,
        analysis_type,
        param_type,
        B_true,
        test_cases=None,
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

        parallel_chains = parser.execution_config['mcmc']['parallel_chains']
        

        # Iterate in sources
        for source, source_results in parser.parsed_results.items():
            param_results = source_results[analysis_type][param_type]
            # Plot chain evolutions
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,20)
            )
            fig.suptitle(
                'Source {} - {}: {}'.format(
                    source,
                    analysis_type,
                    param_type
                ),
                fontsize=25
            )
            for i,j in np.ndindex(B_true.shape):
                for samples in param_results['raw_baseline_samples']:
                    for c in range(parallel_chains):
                        axs[i,j].plot(
                            samples[c][:,i,j],
                            linewidth=.5,
                            # color='cornflowerblue'
                        )
                axs[i,j].set_xlabel(
                    'iteração',
                    fontsize=15
                )
                axs[i,j].axhline(
                    B_true[i,j],
                    linestyle='--',
                    color='gray'
                )
                axs[i,j].set_ylabel(
                    'b{}{}'.format(
                        i+1,
                        j+1
                    ),
                    fontsize=15
                )

            for ax in axs.flat:
                ax.set_axisbelow(True)
                ax.grid(
                    True,
                    which='major',
                    color='lightgray',
                    linestyle='--',
                    linewidth=0.5
                )

            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'trace_plots_{}_{}_{}'.format(analysis_type, param_type, source)))
                    ),
                    bbox_inches='tight'
                )

    @staticmethod
    def IS_trace_plots_OLD(
        results_is,
        B_true,
        test_cases=None,
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

        parallel_chains = 4
        

        for analysis_type, analysis_results in results_is.items():
            # Plot chain evolutions
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,20)
            )
            fig.suptitle(
                analysis_type,
                fontsize=25
            )
            samples_realizations = [analysis_results['raw_baseline_samples']]
            for i,j in np.ndindex(B_true.shape):
                for samples in samples_realizations:
                    for c in range(parallel_chains):
                        axs[i,j].plot(
                            samples[c][:,i,j],
                            # label=c
                            linewidth=.5,
                            color='cornflowerblue'
                        )
                # axs[i,j].legend(
                #     fontsize=15
                # )
                axs[i,j].set_xlabel(
                    'iteração',
                    fontsize=15
                )
                axs[i,j].axhline(
                    B_true[i,j],
                    linestyle='--',
                    color='gray'
                )
                axs[i,j].set_ylabel(
                    'b{}{}'.format(
                        i+1,
                        j+1
                    ),
                fontsize=15
        )
    
    def IS_mcmc_diagnostics(
        parser,
        sources=None,
        return_stats=True,
        whis_lims=(5,95),
        box_color='gray',
        log_scale=False,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        
        def __get_diagnostic(
            diag,
            source,
            analysis_type,
            param_type,
            param_results
        ):
            
            diag_columns = [
                c for c in param_results['mcmc_diagnostics'][0].columns if diag in c
            ]
            source_diagnostics = pd.DataFrame([
                {
                    c: df.iloc[-1,:][c] for c in diag_columns
                } for df in param_results['mcmc_diagnostics']
            ]).melt(
                value_vars=diag_columns,
                var_name='coefficient',
                value_name=diag
            )
            source_diagnostics['source'] = source
            source_diagnostics['analysis_type'] = analysis_type
            source_diagnostics['param_type'] = param_type
            source_diagnostics['coefficient'] = source_diagnostics['coefficient'].str.slice(-4)
            source_diagnostics = source_diagnostics[[
                'source',
                'analysis_type',
                'param_type',
                'coefficient', 
                diag
            ]]

            return source_diagnostics
        

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

        parsed_results = parser.parsed_results

        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }

        # Iterate in sources
        plot_df = pd.DataFrame()
        for source, source_results in parsed_results.items():
            # Iterate in analysis types
            for analysis_type, analysis_results in source_results.items():
                if len(analysis_results)==0:
                    continue
                # Iterate in param types
                for param_type, param_results in analysis_results.items():
                    # Get diagnostics
                    diagnostics = [
                        __get_diagnostic(
                            diag=d,
                            source=source,
                            analysis_type=analysis_type,
                            param_type=param_type,
                            param_results=param_results
                        ) for d in ['r_hat', 'ess']
                    ]

                    # Merge diagnostics dataframes
                    diagnostics_df = diagnostics[0].merge(
                        diagnostics[-1],
                        on=['source', 'coefficient']
                    )
                    
                    plot_df = pd.concat(
                        [
                            plot_df,
                            diagnostics_df
                        ],
                        axis=0
                    )

                    
        # Calculate stats_df
        stats_df = plot_df.groupby(
            by=['source','coefficient'],
            as_index=False
        ).agg(
            r_hat_mean=('r_hat', 'mean'),
            r_hat_std=('r_hat', 'std'),
            r_hat_min=('r_hat', 'min'),
            r_hat_max=('r_hat', 'max'),
            r_hat_median=('r_hat', 'median'),
            ess_mean=('ess', 'mean'),
            ess_std=('ess', 'std'),
            ess_min=('ess', 'min'),
            ess_max=('ess', 'max'),
            ess_median=('ess', 'median')
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(20,15)
        )
        if log_scale:
            ax1.set_yscale('log')
            ax2.set_yscale('log')

        # Plot horizontal line at 1
        ax1.axhline(
            1,
            color='lightgray',
            linestyle='--'
        )

        # Plot horizontal line at r-hat thresh
        ax1.axhline(
            parser.execution_config['mcmc']['R_hat_thresh'],
            color='lightgray',
            linestyle='--'
        )

        sns.boxplot(
            data=plot_df,
            x='source',
            y='r_hat',
            hue='coefficient',
            color=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax1
        )
        ax1.set_ylabel(
            r'$\widehat{R}$'
        )

        sns.boxplot(
            data=plot_df,
            x='source',
            y='ess',
            hue='coefficient',
            color=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax2
        )

        ax2.set_ylabel(
            r'$ESS$'
        )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax1.spines[spine].set_color('black')
            ax2.spines[spine].set_color('black')
        
        ax1.set_xlabel(
            'Distribuição das fontes $r_{s}$'
        )
        ax2.set_xlabel(
            'Distribuição das fontes $r_{s}$'
        )
        
        
        # Plot vertical lines
        [
            ax1.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax1.get_xticks()
        ]
        [
            ax2.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax2.get_xticks()
        ]

        for ax in (ax1, ax2):
            ax.set_axisbelow(True)
            ax.xaxis.grid(False)
            ax.yaxis.grid(
                True,
                color='lightgray',
                linestyle='--',
                linewidth=0.5
            )

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'mcmc_diagnostics'))
                ),
                bbox_inches='tight'
            )
            
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df

    @staticmethod
    def visualize_IS_weights_OLD(
        results_is,
        true_B,
        param_name,
        alpha=0.5,
        n_curves=None,
        log_x=True
    ):

        for analysis_type, analysis_results in results_is.items():
            
            # Get distances from samples to true B's
            dist_true_B = np.array([
                np.linalg.norm(
                    np.subtract(true_B, B_k)
                ) for B_k in analysis_results['merged_baseline_samples']
            ])

            # Get indexes of plotted curves
            if n_curves is not None:
                plotted_idxs = [
                int(idx) for idx in np.linspace(0, len(analysis_results['params'])-1, n_curves)
                ]
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3, ncols=1,
                figsize=(20,25)
            )

            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax3.set_xscale('log')

            fig.suptitle(
                analysis_type,
                fontsize=15
            )
            # Plot weights
            for i, w, param, k_hats in zip(
                range(len(analysis_results['params'])),
                analysis_results['weights'],
                analysis_results['params'],
                analysis_results['k_hat']
            ):
                if i in plotted_idxs:
                    ax1.hist(
                        w,
                        bins='auto',
                        label='{}={}'.format(param_name, f"{param:.3e}"),
                        alpha=alpha,
                        log=True
                    )

                    ax2.scatter(
                        dist_true_B,
                        w,
                        label='{}={}'.format(param_name, f"{param:.3e}"),
                    )
            
            # Plot k-hat      
            ax3.plot(
                analysis_results['params'],
                analysis_results['k_hat']
            )

            ax1.set_xlabel(
                'weight',
                fontsize=15
            )
            ax1.set_ylabel(
                'count',
                fontsize=15
            )
            ax1.legend()

            ax2.set_xlabel(
                '||B-B_k||_F',
                fontsize=15
            )
            ax2.set_ylabel(
                'weight',
                fontsize=15
            )
            ax2.legend()

            ax3.set_xlabel(
                param_name,
                fontsize=15
            )
            ax3.set_ylabel(
                r'$\hat{k}$',
                fontsize=15
            )
            ax3.legend()


    @staticmethod
    def visualize_IS_estimates_OLD(
        results_is,
        true_B,
        param_name
    ):
        def __estimate_plot(
            results,
            var_type,
            analysis_type,
            true_B,
            param_name
        ):
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,15)
            )
            fig.suptitle(
                analysis_type,
                fontsize=15
            )

            for i, j in np.ndindex(true_B.shape):
                axs[i,j].set_xscale('log')
                axs[i,j].plot(
                    results['params'],
                    np.array(results[var_type+'s'])[:,i,j],
                    color='cornflowerblue'
                )
                if var_type=='mean':
                    axs[i,j].axhline(
                        true_B[i,j],
                        color='gray',
                        linestyle='--'
                    )
                axs[i,j].set_xlabel(
                    param_name,
                    fontsize=15
                )
                axs[i,j].set_ylabel(
                    '{}(b_{}{})'.format(
                        var_type,
                        i+1,
                        j+1
                    ),
                    fontsize=15
                )


        for analysis_type, analysis_results in results_is.items():
            _ = [
                __estimate_plot(
                    results=analysis_results,
                    var_type=v,
                    analysis_type=analysis_type,
                    true_B=true_B,
                    param_name=param_name
                ) for v in [
                    'mean',
                    'var',
                    'skew',
                    'kurt'
                ]
            ]

    @staticmethod
    def IS_psis_weights_smoothing(
        parser,
        analysis_type,
        param_type,
        param_values=None,
        sources=None,
        reff=1.0,
        log_y=True,
        cmap='magma',
        return_stats=True,
        save_dir=None,
        save_name=None,
        label_size=15,
        title_size=17,
        legend_size=12
    ):
        def __psis_tail_length(size, reff):
            # Number of weights in the PSIS smoothed tail
            return int(np.ceil(min(size/5, 3*np.sqrt(size/reff))))

        def __get_weights(param_results, idx):
            # Normalized and sorted weights per realization at grid index idx
            norm_sort = lambda w: np.sort(np.asarray(w, dtype=float)/np.sum(w))
            raw_sorted = np.stack([
                norm_sort(rw[idx]) for rw in param_results['raw_weights']
            ])
            smoothed_sorted = np.stack([
                norm_sort(tw[idx]) for tw in param_results['weights']
            ])
            k_hat = np.asarray(param_results['k_hat'], dtype=float)[:, idx]
            return raw_sorted, smoothed_sorted, k_hat

        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }
        colors = [
            to_hex(c) for c in plt.get_cmap(cmap)(np.linspace(0.15, 0.80, len(parsed_results)))
        ]
        # Common perturbation grid (assumed identical across realizations and sources)
        first_results = next(iter(parsed_results.values()))[analysis_type][param_type]
        grid = np.asarray(first_results['params'][0], dtype=float)
        # Evaluated grid points: closest to each requested value, or largest mean k-hat
        if param_values is None:
            mean_k_hat = np.nanmean(np.asarray(first_results['k_hat'], dtype=float), axis=0)
            grid_idxs = [int(np.nanargmax(mean_k_hat))]
        else:
            grid_idxs = [
                int(np.argmin(np.abs(grid - v))) for v in np.atleast_1d(param_values)
            ]
        stats = []
        for idx in grid_idxs:
            param = grid[idx]
            fig, axes = plt.subplots(
                nrows=len(parsed_results), ncols=3,
                figsize=(30, 5*len(parsed_results)),
                squeeze=False
            )
            fig.suptitle(
                'Suavização de Pareto dos pesos de importância: {} - {} = {:.3g} (mediana entre realizações)'.format(
                    analysis_type, param_type, param
                ),
                fontsize=24,
                y=0.995
            )
            for row, ((source, source_results), color) in enumerate(zip(parsed_results.items(), colors)):
                param_results = source_results[analysis_type][param_type]
                raw_sorted, smoothed_sorted, k_hat = __get_weights(param_results, idx)
                n_real, n_samples = raw_sorted.shape
                M = __psis_tail_length(n_samples, reff)
                ranks = np.arange(1, n_samples+1)
                tail_ranks = ranks[-M:]
                # Median order statistic at each position, across realizations
                raw_med = np.median(raw_sorted, axis=0)
                smoothed_med = np.median(smoothed_sorted, axis=0)
                # Ratio computed per realization, then median across realizations
                ratio_med = np.median(
                    smoothed_sorted[:, -M:]/raw_sorted[:, -M:],
                    axis=0
                )
                k_hat_med = np.nanmedian(k_hat)
                stats.append({
                    'param': param,
                    'source': source,
                    'n_realizations': n_real,
                    'n_samples': n_samples,
                    'tail_length': M,
                    'k_hat_median': k_hat_med,
                    'raw_max_weight_median': raw_med[-1],
                    'smoothed_max_weight_median': smoothed_med[-1],
                    'tail_ratio_min_median': np.min(ratio_med)
                })
                ax_w, ax_tail, ax_ratio = axes[row]
                subtitle = r'{} | {}={:.3g} | $\widehat{{k}}$ mediano={:.2f}'.format(
                    source, param_type, param, k_hat_med
                )
                # Left panel: all sorted weights (log-log)
                ax_w.scatter(ranks, raw_med, s=18, color=color, alpha=0.5, label='Pesos brutos')
                ax_w.plot(ranks, smoothed_med, color=color, linewidth=3, label='Pesos suavizados (PSIS)')
                ax_w.axvline(
                    n_samples - M, color='black', linestyle='--', linewidth=1.8,
                    label='Início da cauda suavizada (M={})'.format(M)
                )
                ax_w.set_xscale('log')
                if log_y:
                    ax_w.set_yscale('log')
                ax_w.set_ylabel('Peso normalizado (mediana)', fontsize=label_size)
                ax_w.set_title('Pesos normalizados e ponto de corte\n' + subtitle, fontsize=title_size)
                ax_w.legend(fontsize=legend_size, loc='upper left')
                # Middle panel: smoothed tail only (linear x)
                ax_tail.scatter(tail_ranks, raw_med[-M:], s=24, color=color, alpha=0.5, label='Pesos brutos')
                ax_tail.plot(tail_ranks, smoothed_med[-M:], color=color, linewidth=3, label='Pesos suavizados (PSIS)')
                if log_y:
                    ax_tail.set_yscale('log')
                ax_tail.set_ylabel('Peso normalizado (mediana)', fontsize=label_size)
                ax_tail.set_title('Pesos na cauda suavizada\n' + subtitle, fontsize=title_size)
                ax_tail.legend(fontsize=legend_size, loc='upper left')
                # Right panel: smoothed / raw ratio on the tail
                ax_ratio.axhline(1, color='lightgray', linestyle='--')
                ax_ratio.plot(tail_ranks, ratio_med, color=color, linewidth=2.5)
                ax_ratio.set_ylabel('Suavizado / bruto (mediana)', fontsize=label_size)
                ax_ratio.set_title('Efeito da suavização na cauda\n' + subtitle, fontsize=title_size)
                for ax in (ax_w, ax_tail, ax_ratio):
                    ax.set_xlabel('Posição a partir do menor peso', fontsize=label_size)
                    ax.tick_params(axis='both', labelsize=13)
                    ax.grid(alpha=0.25, linestyle='--')
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}_{}_{:.3g}.png'.format(
                            save_name if save_name is not None else 'psis_weights_smoothing_{}'.format(analysis_type),
                            param_type,
                            param
                        ))
                    ),
                    bbox_inches='tight'
                )
        if return_stats:
            return pd.DataFrame(stats)
            

    @staticmethod
    def IS_psis_max_k_hat_diagnostics(
        parser,
        analysis_type,
        param_type,
        sources=None,
        k_hat_thresh=0.7,
        return_stats=True,
        whis_lims=(5,95),
        box_color='gray',
        log_scale=False,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        def __get_max_k_hat(
            source,
            param_results
        ):
            rows = []
            for r, (params, k_hat) in enumerate(
                zip(
                    param_results['params'],
                    param_results['k_hat']
                )
            ):
                params = np.asarray(params, dtype=float)
                k_hat = np.asarray(k_hat, dtype=float)
                # Ignore non-finite k-hat values (e.g. failed Pareto fits)
                valid = np.isfinite(k_hat)
                if not valid.any():
                    continue
                idx_max = np.argmax(np.where(valid, k_hat, -np.inf))
                rows.append({
                    'source': source,
                    'realization': r,
                    'k_hat_max': k_hat[idx_max],
                    'param_at_max': params[idx_max]
                })
            return pd.DataFrame(rows)

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
        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }
        # Iterate in sources: max k-hat over the perturbation grid, per realization
        plot_df = pd.concat(
            [
                __get_max_k_hat(
                    source=source,
                    param_results=source_results[analysis_type][param_type]
                ) for source, source_results in parsed_results.items()
            ],
            axis=0
        ).reset_index(
            drop=True
        )
        # Calculate stats_df
        stats_df = plot_df.groupby(
            by='source',
            as_index=False,
            sort=False
        ).agg(
            k_hat_max_mean=('k_hat_max', 'mean'),
            k_hat_max_std=('k_hat_max', 'std'),
            k_hat_max_min=('k_hat_max', 'min'),
            k_hat_max_median=('k_hat_max', 'median'),
            k_hat_max_max=('k_hat_max', 'max'),
            k_hat_max_q05=('k_hat_max', lambda v: np.quantile(v, .05)),
            k_hat_max_q95=('k_hat_max', lambda v: np.quantile(v, .95)),
            frac_above_thresh=('k_hat_max', lambda v: np.mean(v > k_hat_thresh)),
            n_realizations=('k_hat_max', 'count')
        )
        # Create plot
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,7)
        )
        fig.suptitle(
            r'Máximo de $\hat{{k}}$ no espaço de perturbação: {} - {}'.format(
                analysis_type,
                param_type
            ),
            fontsize=label_size
        )
        if log_scale:
            ax.set_yscale('log')
        # Pareto k-hat horizontal lines at 0 and thresh
        [
            ax.axhline(
                v,
                color='lightgray',
                linestyle='--'
            ) for v in [0, k_hat_thresh]
        ]
        sns.boxplot(
            data=plot_df,
            x='source',
            y='k_hat_max',
            order=list(parsed_results.keys()),
            color=box_color,
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax
        )
        ax.set_ylabel(
            r'$\max_{\xi}\,\hat{k}$'
        )
        ax.set_xlabel(
            'Distribuição das fontes $r_{s}$'
        )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_color('black')
        # Plot vertical lines
        [
            ax.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax.get_xticks()
        ]

        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

        # Save figure, if so specified
        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'psis_max_k_hat_{}_{}'.format(analysis_type, param_type)))
                ),
                bbox_inches='tight'
            )
        # Calculate statistics, if so specified
        if return_stats:
            return stats_df
        

    @staticmethod
    def visualize_IS_estimates(
        results_is,
        true_B,
        param_name,
        log_x=True,
        cmap=None
    ):
        def __estimate_plot(
            results,
            var_type,
            true_B,
            param_name,
            cmap
        ):
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,15)
            )
            fig.suptitle(
                'Variation: {}'.format(param_name),
                fontsize=25
            )
            for i, j in np.ndindex(true_B.shape):
                if log_x:
                    axs[i,j].set_xscale('log')
                for k, (analysis_type, analysis_results) in enumerate(results.items()): 
                    axs[i,j].plot(
                        analysis_results['params'],
                        np.array(analysis_results[var_type+'s'])[:,i,j],
                        color=cmap[k],
                        label=analysis_type
                    )
                    if var_type=='mean':
                        axs[i,j].axhline(
                            true_B[i,j],
                            color='gray',
                            linestyle='--'
                        )
                    axs[i,j].set_xlabel(
                        param_name,
                        fontsize=15
                    )
                    axs[i,j].set_ylabel(
                        '{}(b_{}{})'.format(
                            var_type,
                            i+1,
                            j+1
                        ),
                        fontsize=15
                    )
                
                # Plot legend
                axs[i,j].legend()

        # Get colormap
        if cmap is None:
            cmap = plt.cm.copper(
                np.linspace(0,1,len(results_is))
            )
        else:
            cmap = cmap(np.linspace(0,1,len(results_is)))
        
        # Create plots
        _ = [
            __estimate_plot(
                results=results_is,
                var_type=v,
                true_B=true_B,
                param_name=param_name,
                cmap=cmap
            ) for v in [
                'mean',
                'var',
                'skew',
                'kurt'
            ]
        ]

    @staticmethod
    def visualize_IS_weights(
        parser,
        analysis_type,
        param_type,
        true_B,
        sources=None,
        alpha=0.5,
        n_curves=None,
        cmap='viridis',
        log_x=True,
        label_size=20,
        legend_size=20
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

        parsed_results = parser.parsed_results

        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }

        _cmap = plt.get_cmap(cmap)

        for source, source_results in parsed_results.items():

            # Retrieve results for analysis type and parameter
            param_results = source_results[analysis_type][param_type]
            
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3, ncols=1,
                figsize=(20,25)
            )

            # Get indexes of plotted curves
            if n_curves is not None:
                plotted_idxs = [
                    int(idx) for idx in np.linspace(0, len(param_results['params'][0])-1, n_curves)
                ]
            else:
                plotted_idxs = range(len(param_results['params'][0]))
            
            # Instantiate colormaps
            colors = _cmap(
                np.linspace(0.1, 0.9,len(plotted_idxs))
            )
            if log_x:
                ax3.set_yscale('log')
                ax3.set_xscale('log')

            fig.suptitle(
                '{} - {}: {}'.format(
                    source,
                    analysis_type,
                    param_type
                ),
                fontsize=label_size
            )

            # Raw weights: ax1
            for r, (realization_weights, params) in enumerate(
                zip(
                    param_results['raw_weights'],
                    param_results['params']
                )
            ):
                print(
                    np.mean(
                        np.stack(param_results['raw_weights']),
                        axis=0
                    ).shape
                )
                raise
                [
                    ax1.plot(
                        range(len(rw)),
                        np.cumsum(rw[np.argsort(rw)])/np.sum(rw),
                        color=colors[i],
                        # linestyle='--',
                        # linewidth=.5,
                        label=None if r > 0 else round(p,3)
                    ) for i, rw, p in zip(
                        range(len(plotted_idxs)),
                        np.asarray(realization_weights)[plotted_idxs],
                        np.asarray(params)[plotted_idxs]
                    )
                ]
            
            # Adjusted weights: ax2
            for r, (realization_weights, params) in enumerate(
                zip(
                    param_results['weights'],
                    param_results['params']
                )
            ):
                [
                    ax2.plot(
                        range(len(rw)),
                        np.cumsum(rw[np.argsort(rw)])/np.sum(rw),
                        color=colors[i],
                        # linestyle='--',
                        # linewidth=.5,
                        label=None if r > 0 else '{}: {}'.format(param_type, round(p,3))
                    ) for i, rw, p in zip(
                        range(len(plotted_idxs)),
                        np.asarray(realization_weights)[plotted_idxs],
                        np.asarray(params)[plotted_idxs]
                    )
                ]

            # Raw weights vs adjusted weights: ax3
            for r, (raw_weights, transformed_weights, params) in enumerate(
                zip(
                    param_results['raw_weights'],
                    param_results['weights'],
                    param_results['params']
                )
            ):
                [
                    ax3.plot(
                        rw,
                        tw,
                        color=colors[i],
                        # linestyle='--',
                        # linewidth=.5,
                        label=None if r > 0 else '{}: {}'.format(param_type, round(p,3))
                    ) for i, rw, tw, p in zip(
                        range(len(plotted_idxs)),
                        np.asarray(raw_weights)[plotted_idxs],
                        np.asarray(transformed_weights)[plotted_idxs],
                        np.asarray(params)[plotted_idxs]
                    )
                ]

            ax1.legend()
            ax2.legend()

            # ax1.set_xlabel(
            #     ''
            # )
    

    @staticmethod
    def visualize_average_IS_weights(
        parser,
        analysis_type,
        param_type,
        true_B,
        sources=None,
        alpha=0.5,
        n_curves=None,
        cmap='viridis',
        log_x=True,
        label_size=20,
        legend_size=20,
        save_dir=None,
        save_name=None
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

        parsed_results = parser.parsed_results

        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }

        _cmap = plt.get_cmap(cmap)

        for source, source_results in parsed_results.items():

            # Retrieve results for analysis type and parameter
            param_results = source_results[analysis_type][param_type]
            
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3, ncols=1,
                figsize=(20,25)
            )

            # Get indexes of plotted curves
            if n_curves is not None:
                plotted_idxs = [
                    int(idx) for idx in np.linspace(0, len(param_results['params'][0])-1, n_curves)
                ]
            else:
                plotted_idxs = range(len(param_results['params'][0]))
            
            # Instantiate colormaps
            colors = _cmap(
                np.linspace(0.1, 0.9,len(plotted_idxs))
            )

            fig.suptitle(
                '{} - {}: {}'.format(
                    source,
                    analysis_type,
                    param_type
                ),
                fontsize=label_size
            )

            # Function to return normalized cumulative sum of sorted weights
            sorted_cumsum_fn = lambda w: np.cumsum(
                w[np.argsort(w)]
            )/np.sum(w)

            # Raw weights: ax1
            sorted_cumsums_rw = np.stack(
                [
                    [
                        sorted_cumsum_fn(w) for w in realization_weights
                    ] for realization_weights in param_results['raw_weights']
                ] 
            )
            mean_sorted_cumsums_rw = np.mean(
                sorted_cumsums_rw,
                axis=0
            )
            [
                ax1.plot(
                    np.asarray(range(sorted_cumsums_rw.shape[-1]))/sorted_cumsums_rw.shape[-1],
                    sorted_cumsum,
                    color=colors[i],
                    label='{}={}'.format(param_type, round(p,3))
                ) for i, (p, sorted_cumsum) in enumerate(
                    zip(
                        np.array(param_results['params'][0])[plotted_idxs],
                        mean_sorted_cumsums_rw[plotted_idxs]
                    )
                )
            ]
            ax1.set_xlabel(
                'k/K',
                fontsize=label_size
            )
            ax1.set_title(
                'Soma cumulativa normalizada - Pesos IS Não-tratados',
                fontsize=label_size
            )
            ax1.legend(fontsize=legend_size)

            
            
            # Adjusted weights: ax2
            sorted_cumsums_tw = np.stack(
                [
                    [
                        sorted_cumsum_fn(w) for w in realization_weights
                    ] for realization_weights in param_results['weights']
                ] 
            )
            mean_sorted_cumsums_tw = np.mean(
                sorted_cumsums_tw,
                axis=0
            )

            [
                ax2.plot(
                    np.asarray(range(sorted_cumsums_tw.shape[-1]))/sorted_cumsums_tw.shape[-1],
                    sorted_cumsum,
                    color=colors[i],
                    label='{}={}'.format(param_type, round(p,3))
                ) for i, (p, sorted_cumsum) in enumerate(
                    zip(
                        np.array(param_results['params'][0])[plotted_idxs],
                        mean_sorted_cumsums_tw[plotted_idxs]
                    )
                )
            ]
            ax2.set_xlabel(
                'k/K',
                fontsize=label_size
            )
            ax2.set_title(
                'Soma cumulativa normalizada - Pesos IS Suavizados',
                fontsize=label_size
            )
            ax2.legend(fontsize=legend_size)

            # Raw weights vs adjusted weights: ax3
            [
                ax3.plot(
                    rw,
                    tw,
                    color=colors[i],
                    label='{}={}'.format(param_type, round(p,3))
                ) for i, (p, rw, tw) in enumerate(
                    zip(
                        np.array(param_results['params'][0])[plotted_idxs],
                        mean_sorted_cumsums_rw[plotted_idxs],
                        mean_sorted_cumsums_tw[plotted_idxs]
                    )
                )
            ]
            ax3.set_xlabel(
                'Pesos não-tratados',
                fontsize=label_size
            )
            ax3.set_ylabel(
                'Pesos suavizados',
                fontsize=label_size
            )
            ax3.set_title(
                'Somas cumulativas normalizadas: Pesos não-tratados vs Pesos IS Suavizados',
                fontsize=label_size
            )
            ax3.legend(fontsize=legend_size)

            ax2.set_xlabel(
                'k/K',
                fontsize=label_size
            )
            ax2.set_title(
                'Soma cumulativa normalizada - Pesos IS Suavizados',
                fontsize=label_size
            )
            ax2.legend(fontsize=legend_size)

            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'average_IS_weights_{}_{}_{}'.format(analysis_type, param_type, source)))
                    ),
                    bbox_inches='tight'
                )

    # @staticmethod
    # def visualize_average_IS_estimates(
    #     parser,
    #     analysis_type,
    #     param_type,
    #     true_B,
    #     sources=None,
    #     alpha=0.5,
    #     n_curves=None,
    #     cmap='viridis',
    #     log_x=True,
    #     label_size=20,
    #     legend_size=20,
    #     save_dir=None,
    #     save_name=None
    # ):
    #     def __estimate_plot(
    #         results,
    #         var_type,
    #         true_B,
    #         analysis_type,
    #         param_type,
    #         cmap,
    #         save_dir,
    #         save_name
    #     ):
    #         fig, axs = plt.subplots(
    #             nrows=2, ncols=2,
    #             figsize=(25,15)
    #         )
    #         fig.suptitle(
    #             'Analysis: {} - {}'.format(analysis_type, param_type),
    #             fontsize=25
    #         )
    #         for k, (source, source_results) in enumerate(results.items()):
    #             # Retrieve results for analysis and parameter type
    #             param_results = source_results[analysis_type][param_type]

    #             # Retrieve average results for variable type
    #             average_var_curves = np.mean(
    #                 np.stack(param_results[var_type+'s']),
    #                 axis=0
    #             )

    #             for i, j in np.ndindex(true_B.shape):
    #                 if log_x:
    #                     axs[i,j].set_xscale('log')
                    
    #                 axs[i,j].plot(
    #                     param_results['params'][0],
    #                     average_var_curves[:,i,j],
    #                     color=cmap[k],
    #                     label=source
    #                 )
    #                 if var_type=='mean':
    #                     axs[i,j].axhline(
    #                         true_B[i,j],
    #                         color='gray',
    #                         linestyle='--'
    #                     )
    #                 axs[i,j].set_xlabel(
    #                     param_type,
    #                     fontsize=15
    #                 )
    #                 axs[i,j].set_ylabel(
    #                     '{}(b_{}{})'.format(
    #                         var_type,
    #                         i+1,
    #                         j+1
    #                     ),
    #                     fontsize=15
    #                 )
                    
    #                 # Plot legend
    #                 axs[i,j].legend()
            
    #         for ax in axs.flat:
    #             ax.set_axisbelow(True)
    #             ax.grid(
    #                 True,
    #                 which='major',
    #                 color='lightgray',
    #                 linestyle='--',
    #                 linewidth=0.5
    #             )

    #         # Save figure, if so specified
    #         if save_dir is not None:
    #             fig.savefig(
    #                 str(
    #                     (save_dir / '{}.png'.format(save_name if save_name is not None else 'average_IS_estimates_{}_{}_{}'.format(analysis_type, param_type, var_type)))
    #                 ),
    #                 bbox_inches='tight'
    #             )

    #     plt.rc(
    #         'legend',
    #         fontsize=legend_size,
    #         title_fontsize=legend_size
    #     )

    #     plt.rcParams['axes.facecolor'] = 'white'
    #     plt.rcParams['savefig.facecolor'] = 'white'

    #     plt.rc('axes', labelsize=label_size)
    #     plt.rc('xtick', labelsize=label_size)
    #     plt.rc('ytick', labelsize=label_size)

    #     parsed_results = parser.parsed_results

    #     if sources is not None:
    #         parsed_results = {
    #             k: v for k, v in parsed_results.items() if k in sources
    #         }

        

    #     # Get colormap
    #     if cmap is None:
    #         _cmap = plt.cm.copper(
    #             np.linspace(0.1,0.9,len(parsed_results))
    #         )
    #     else:
    #         _cmap = plt.get_cmap(cmap)
    #         _cmap = _cmap(np.linspace(0.1, 0.9, len(parsed_results)))

        
    #     # Create plots
    #     _ = [
    #         __estimate_plot(
    #             results=parsed_results,
    #             var_type=v,
    #             true_B=true_B,
    #             analysis_type=analysis_type,
    #             param_type=param_type,
    #             cmap=_cmap,
    #             save_dir=save_dir,
    #             save_name=save_name
    #         ) for v in [
    #             'mean',
    #             'var',
    #             'skew',
    #             'kurt'
    #         ]
    #     ]

    @staticmethod
    def visualize_IS_moments(
        parser,
        analysis_type,
        param_type,
        true_B,
        sources=None,
        alpha=0.5,
        n_curves=None,
        cmap='magma',
        log_x=True,
        label_size=20,
        legend_size=20,
        save_dir=None,
        save_name=None,
        param_name=None,
        ci_prob=0.95,
        n_sems=None
    ):
        def __mean_band(values, ci_prob, n_sems):
            # Mean across realizations (axis 0) and its uncertainty band:
            # mean ± z * SEM, with SEM = sample std / sqrt(R)
            from statistics import NormalDist
            n = values.shape[0]
            mean = np.mean(values, axis=0)
            if n < 2:
                return mean, mean, mean
            sem = np.std(values, axis=0, ddof=1)/np.sqrt(n)
            z = n_sems if n_sems is not None else NormalDist().inv_cdf(0.5 + ci_prob/2)
            return mean, mean - z*sem, mean + z*sem

        def __estimate_plot(
            results,
            var_type,
            true_B,
            analysis_type,
            param_type,
            cmap,
            save_dir,
            save_name
        ):
            fig, axs = plt.subplots(
                nrows=2, ncols=2,
                figsize=(25,15)
            )
            fig.suptitle(
                'Analysis: {} - {} - {} ({} entre realizações)'.format(
                    analysis_type, param_type, var_type, band_label
                ),
                fontsize=label_size
            )
            for k, (source, source_results) in enumerate(results.items()):
                # Retrieve results for analysis and parameter type
                param_results = source_results[analysis_type][param_type]
                x = np.asarray(param_results['params'][0], dtype=float)
                # Formato: (realizações, pontos da varredura, n, n)
                values = np.asarray(
                    np.stack(param_results[var_type+'s']),
                    dtype=float
                )
                mean, band_low, band_high = __mean_band(
                    values, ci_prob=ci_prob, n_sems=n_sems
                )
                for i, j in np.ndindex(true_B.shape):
                    ax = axs[i,j]
                    # Mean line
                    ax.plot(
                        x,
                        mean[:,i,j],
                        color=cmap[k],
                        linewidth=5,
                        label=source
                    )
                    # Band bounds
                    ax.plot(
                        x,
                        band_low[:,i,j],
                        color=cmap[k],
                        linestyle='--'
                    )
                    ax.plot(
                        x,
                        band_high[:,i,j],
                        color=cmap[k],
                        linestyle='--'
                    )
                    # Band fill
                    ax.fill_between(
                        x,
                        band_low[:,i,j],
                        band_high[:,i,j],
                        color=cmap[k],
                        alpha=0.2
                    )
            for i, j in np.ndindex(true_B.shape):
                ax = axs[i,j]
                if log_x:
                    ax.set_xscale('log')
                if var_type=='mean':
                    ax.axhline(
                        true_B[i,j],
                        color='gray',
                        linestyle='--'
                    )
                ax.set_xlabel(
                    param_type if param_name is None else param_name,
                    fontsize=label_size
                )
                ax.set_ylabel(
                    '{}(b_{}{})'.format(
                        var_type,
                        i+1,
                        j+1
                    ),
                    fontsize=label_size
                )
                ax.legend(
                    fontsize=legend_size
                )
                ax.set_axisbelow(True)
                ax.grid(False, which='both')
                ax.grid(
                    True,
                    which='major',
                    axis='both',
                    color='gray',
                    linestyle='--',
                    linewidth=0.5,
                    alpha=0.25,
                )
            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}.png'.format(save_name if save_name is not None else 'average_IS_estimates_{}_{}_{}'.format(analysis_type, param_type, var_type)))
                    ),
                    bbox_inches='tight'
                )

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
        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }
        # Get colormap
        if cmap is None:
            _cmap = plt.cm.copper(
                np.linspace(0.1,0.9,len(parsed_results))
            )
        else:
            _cmap = plt.get_cmap(cmap)
            _cmap = _cmap(np.linspace(0.1, 0.9, len(parsed_results)))
        band_label = (
            r'média $\pm$ {:g} SEM'.format(n_sems) if n_sems is not None
            else 'média e IC {:.0f}% da média (SEM)'.format(100*ci_prob)
        )
        # Create plots
        for v in ['mean', 'var', 'skew', 'kurt']:
            __estimate_plot(
                results=parsed_results,
                var_type=v,
                true_B=true_B,
                analysis_type=analysis_type,
                param_type=param_type,
                cmap=_cmap,
                save_dir=save_dir,
                save_name=save_name
            )
                     


    @staticmethod
    def IS_errors_amplitudes(
        parser,
        analysis_type,
        param_type,
        true_B,
        sources=None,
        param_name=None,
        whis_lims=(5,95),
        box_color='gray',
        log_scale=False,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20,
        cmap='magma',
        n_printed_stds=3,
        ci_prob=0.95,
        n_sems=None
    ):
        def __get_variations_and_errors(
            cfg,
            param_results,
            source,
            analysis_type,
            param_type
        ):
            # Function for calculating B estimation error
            B_error_fn = lambda B: np.linalg.norm(
                np.subtract(
                    cfg['general']['B'],
                    B
                )
            )/np.linalg.norm(
                cfg['general']['B']
            )
            # Function for calculating source estimation error
            s_error_fn = lambda s, x, B: np.linalg.norm(
                np.subtract(
                    s,
                    B@x
                )
            )/np.linalg.norm(s)
            # Function for calculating each coefficient's amplitude
            ampl_fn = lambda Bs: np.divide(
                np.subtract(
                    np.max(Bs, axis=0),
                    np.min(Bs, axis=0)
                ),
                np.abs(cfg['general']['B'])
            )
            # Get estimated matrices
            B_est = np.array(param_results['means'])
            # Get B errors
            B_errors = np.array([
                [
                    B_error_fn(B=B) for B in realization_Bs
                ] for realization_Bs in B_est
            ])
            # Get s errors
            s_errors = np.array([
                [
                    s_error_fn(
                        s=realization_s,
                        x=realization_x,
                        B=B
                    ) for B in realization_Bs
                ] for realization_s, realization_x, realization_Bs in zip(
                    param_results['s'],
                    param_results['x'],
                    B_est
                )
            ])
            # Get amplitudes
            amplitudes = np.array([
                ampl_fn(realization_Bs) for realization_Bs in B_est
            ])
            return {
                'B_errors': B_errors,
                's_errors': s_errors,
                'amplitudes': amplitudes
            }

        def __mean_band(values, ci_prob, n_sems):
            # Mean across realizations (axis 0) and its uncertainty band:
            # mean ± z * SEM, with SEM = sample std / sqrt(R)
            from statistics import NormalDist
            n = values.shape[0]
            mean = np.mean(values, axis=0)
            if n < 2:
                return mean, mean, mean
            sem = np.std(values, axis=0, ddof=1)/np.sqrt(n)
            z = n_sems if n_sems is not None else NormalDist().inv_cdf(0.5 + ci_prob/2)
            return mean, mean - z*sem, mean + z*sem

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
        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }
            sources = np.asarray(parsed_results.keys())
        # Get colormap
        if cmap is None:
            _cmap = plt.cm.copper(
                np.linspace(0.1,0.9,len(parsed_results))
            )
        else:
            _cmap = plt.get_cmap(cmap)
            _cmap = _cmap(np.linspace(0.1, 0.9, len(parsed_results)))
        fig1, (ax1_1, ax1_2) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(20,20)
        )
        fig2, ax2_1 = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,7)
        )
        if log_scale:
            ax1_1.set_xscale('log')
            ax1_2.set_xscale('log')
        band_label = (
            r'média $\pm$ {:g} SEM'.format(n_sems) if n_sems is not None
            else 'média e IC {:.0f}% da média (SEM)'.format(100*ci_prob)
        )
        fig1.suptitle(
            'Recovery errors analysis: {} - {} ({} entre realizações)'.format(
                analysis_type, param_type, band_label
            ),
            fontsize=label_size
        )
        fig2.suptitle(
            'Amplitudes distributions analysis: {} - {}'.format(analysis_type, param_type),
            fontsize=label_size
        )
        ax1_1.set_xlabel(
            param_type if param_name is None else param_name,
            fontsize=label_size
        )
        ax1_2.set_xlabel(
            param_type if param_name is None else param_name,
            fontsize=label_size
        )
        ax1_1.set_ylabel(
            r'$\delta_{\bf{B}}(\widehat{\bf{B}})$',
            fontsize=label_size
        )
        ax1_2.set_ylabel(
            r'$\delta_{\bf{S}}(\widehat{\bf{S}})$',
            fontsize=label_size
        )
        for ax in fig1.axes:
            ax.set_axisbelow(True)
            ax.grid(False, which='both')
            ax.grid(
                True,
                which='major',
                axis='both',
                color='gray',
                linestyle='--',
                linewidth=0.5,
                alpha=0.25,
            )
        ax2_1.set_axisbelow(True)
        ax2_1.grid(False, which='both')
        ax2_1.grid(
            True,
            axis='y',
            which='major',
            linestyle='--',
            linewidth=0.5,
            alpha=0.25,
        )
        errors = {}
        amplitudes = pd.DataFrame()
        for k, (source, source_results) in enumerate(parsed_results.items()):
            # Retrieve results for analysis type and parameter
            param_results = source_results[analysis_type][param_type]
            # Measure sensitivities
            results = __get_variations_and_errors(
                cfg=parser.execution_config,
                param_results=param_results,
                source=source,
                analysis_type=analysis_type,
                param_type=param_type
            )
            errors[source] = results
            x = np.asarray(param_results['params'][0], dtype=float)
            for ax, key in (
                (ax1_1, 'B_errors'),
                (ax1_2, 's_errors'),
            ):
                # Formato: (realizações, pontos da varredura)
                values = np.asarray(results[key], dtype=float)
                mean, band_low, band_high = __mean_band(
                    values, ci_prob=ci_prob, n_sems=n_sems
                )
                # Mean line
                ax.plot(
                    x,
                    mean,
                    color=_cmap[k],
                    linewidth=5,
                    label=source
                )
                # Band bounds
                ax.plot(
                    x,
                    band_low,
                    color=_cmap[k],
                    linestyle='--'
                )
                ax.plot(
                    x,
                    band_high,
                    color=_cmap[k],
                    linestyle='--'
                )
                # Band fill
                ax.fill_between(
                    x,
                    band_low,
                    band_high,
                    color=_cmap[k],
                    alpha=0.2
                )
            # Plot coefficient amplitudes
            source_amplitudes = pd.DataFrame()
            for i, j in np.ndindex(true_B.shape):
                source_amplitudes = pd.concat(
                    [
                        source_amplitudes,
                        pd.DataFrame(
                            data={
                                'source': [source]*results['amplitudes'].shape[0],
                                'coefficient': ['b{}{}'.format(i+1,j+1)]*results['amplitudes'].shape[0],
                                'amplitude': results['amplitudes'][:,i,j]
                            }
                        )
                    ],
                    axis=0
                )
            amplitudes = pd.concat(
                [
                    amplitudes,
                    source_amplitudes
                ],
                axis=0
            ).reset_index(
                drop=True
            )
        ax1_1.legend(
            fontsize=legend_size
        )
        ax1_2.legend(
            fontsize=legend_size
        )
        sns.boxplot(
            data=amplitudes,
            x='source',
            y='amplitude',
            hue='coefficient',
            palette='dark:{}'.format(box_color),
            whis=whis_lims,
            width=.7,
            showfliers=False,
            ax=ax2_1
        )
        for spine in ['bottom', 'top', 'left', 'right']:
            ax2_1.spines[spine].set_color('black')
        ax2_1.set_xlabel(
            'Distribuição das fontes $r_{s}$'
        )
        # Plot vertical lines
        [
            ax2_1.axvline(
                x+.5,
                color='lightgray',
                linestyle='--'
            ) for x in ax2_1.get_xticks()
        ]
        # Save figure, if so specified
        if save_dir is not None:
            fig1.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'recovery_errors_{}_{}'.format(analysis_type, param_type)))
                ),
                bbox_inches='tight'
            )
            fig2.savefig(
                str(
                    (save_dir / '{}.png'.format(save_name if save_name is not None else 'amplitudes_{}_{}'.format(analysis_type, param_type)))
                ),
                bbox_inches='tight'
            )
        # Standard errors of the mean across independent realizations.
        # Expected arrays: (realizations, parameter points).
        if (
            n_printed_stds is not None
            and (
                isinstance(n_printed_stds, (bool, np.bool_))
                or not isinstance(n_printed_stds, (int, np.integer))
                or n_printed_stds < 1
            )
        ):
            raise ValueError(
                'n_printed_stds must be a positive integer '
                'or None to print all rows.'
            )
        for metric in ('B_errors', 's_errors'):
            columns = {}
            for source, source_errors in errors.items():
                param_results = (
                    parsed_results[source][analysis_type][param_type]
                )
                grids = np.asarray(
                    param_results['params'],
                    dtype=float,
                )
                values = np.asarray(
                    source_errors[metric],
                    dtype=float,
                )
                # Validate the realization-by-parameter structure.
                if values.ndim != 2:
                    raise ValueError(
                        f'{source}/{metric}: expected a 2D array '
                        f'(realizations, parameter points), '
                        f'got shape {values.shape}.'
                    )
                if grids.shape != values.shape:
                    raise ValueError(
                        f'{source}/{metric}: parameter grids and errors '
                        f'must have matching shapes; got '
                        f'{grids.shape} and {values.shape}.'
                    )
                n_realizations, n_points = values.shape
                if n_realizations == 0 or n_points == 0:
                    raise ValueError(
                        f'{source}/{metric}: no observations available.'
                    )
                if not np.isfinite(grids).all():
                    raise ValueError(
                        f'{source}: non-finite parameter values found.'
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f'{source}/{metric}: non-finite errors found.'
                    )
                # Each parameter column must represent the same point
                # across all realizations.
                if not np.allclose(
                    grids,
                    grids[0],
                    rtol=1e-12,
                    atol=1e-14,
                ):
                    raise ValueError(
                        f'{source}: parameter grids differ '
                        'between realizations.'
                    )
                # SEM = sample standard deviation / sqrt(realization count).
                if n_realizations < 2:
                    sem = np.full(n_points, np.nan)
                else:
                    sem = (
                        np.std(values, axis=0, ddof=1)
                        / np.sqrt(n_realizations)
                    )
                index = pd.Index(
                    grids[0],
                    name=param_type,
                )
                if not index.is_unique:
                    raise ValueError(
                        f'{source}: duplicate parameter points found.'
                    )
                columns[source] = pd.Series(
                    sem,
                    index=index,
                )
            # Align sources by their actual parameter values.
            table = pd.DataFrame(columns).sort_index()
            table.columns.name = 'r_s'
            # Select a limited number of rows for printing only.
            if n_printed_stds is None:
                printed_table = table
            else:
                n_rows = min(n_printed_stds, len(table))
                row_indices = np.linspace(
                    0,
                    len(table) - 1,
                    num=n_rows,
                    dtype=int,
                )
                printed_table = table.iloc[row_indices]
            print(
                '\n{}: standard error of the mean across realizations'.format(metric) +
                f' | {analysis_type} | {param_type} | {metric}'
                f' | showing {len(printed_table)}/{len(table)} points'
            )
            print(
                printed_table.to_string(
                    float_format=lambda value: f'{value:.6e}',
                    na_rep='NaN',
                )
            )

    # @staticmethod
    # def visualize_sensitivities(
    #     parser,
    #     analysis_type,
    #     param_type,
    #     true_B,
    #     sources=None,
    #     param_name=None,
    #     return_stats=True,
    #     whis_lims=(5,95),
    #     box_color='gray',
    #     log_scale=False,
    #     save_dir=None,
    #     save_name=None,
    #     label_size=20,
    #     legend_size=20,
    #     cmap='magma',
    #     n_printed_stds=3
    # ):
    #     def __get_variations_and_errors(
    #         cfg,
    #         param_results,
    #         source,
    #         analysis_type,
    #         param_type
    #     ):
    #         # Function for calculating B estimation error
    #         B_error_fn = lambda B: np.linalg.norm(
    #             np.subtract(
    #                 cfg['general']['B'],
    #                 B
    #             )
    #         )/np.linalg.norm(
    #             cfg['general']['B']
    #         )

    #         # Function for calculating source estimation error
    #         s_error_fn = lambda s, x, B: np.linalg.norm(
    #             np.subtract(
    #                 s,
    #                 B@x
    #             )
    #         )/np.linalg.norm(s)

    #         # Function for calculating each coefficient's amplitude
    #         ampl_fn = lambda Bs: np.divide(
    #             np.subtract(
    #                 np.max(Bs, axis=0),
    #                 np.min(Bs, axis=0)
    #             ),
    #             np.abs(cfg['general']['B'])
    #         )

    #         # Get estimated matrices
    #         B_est = np.array(param_results['means'])

    #         # Get B errors
    #         B_errors = np.array([
    #             [
    #                 B_error_fn(B=B) for B in realization_Bs
    #             ] for realization_Bs in B_est
    #         ])

    #         # Get s errors
    #         s_errors = np.array([
    #             [
    #                 s_error_fn(
    #                     s=realization_s,
    #                     x=realization_x,
    #                     B=B
    #                 ) for B in realization_Bs
    #             ] for realization_s, realization_x, realization_Bs in zip(
    #                 param_results['s'],
    #                 param_results['x'],
    #                 B_est
    #             )
    #         ])

    #         # Get amplitudes
    #         amplitudes = np.array([
    #             ampl_fn(realization_Bs) for realization_Bs in B_est
    #         ])

    #         return {
    #             'B_errors': B_errors,
    #             's_errors': s_errors,
    #             'amplitudes': amplitudes
    #         }
            


    #     plt.rc(
    #         'legend',
    #         fontsize=legend_size,
    #         title_fontsize=legend_size
    #     )

    #     plt.rcParams['axes.facecolor'] = 'white'
    #     plt.rcParams['savefig.facecolor'] = 'white'

    #     plt.rc('axes', labelsize=label_size)
    #     plt.rc('xtick', labelsize=label_size)
    #     plt.rc('ytick', labelsize=label_size)

    #     parsed_results = parser.parsed_results

    #     if sources is not None:
    #         parsed_results = {
    #             k: v for k, v in parsed_results.items() if k in sources
    #         }
    #         sources = np.asarray(parsed_results.keys())
            
    #     # Get colormap
    #     if cmap is None:
    #         _cmap = plt.cm.copper(
    #             np.linspace(0.1,0.9,len(parsed_results))
    #         )
    #     else:
    #         _cmap = plt.get_cmap(cmap)
    #         _cmap = _cmap(np.linspace(0.1, 0.9, len(parsed_results)))

    #     fig1, (ax1_1, ax1_2) = plt.subplots(
    #         nrows=2, ncols=1,
    #         figsize=(20,20)
    #     )

    #     fig2, ax2_1 = plt.subplots(
    #         nrows=1, ncols=1,
    #         figsize=(20,7)
    #     )

    #     if log_scale:
    #         ax1_1.set_xscale('log')
    #         ax1_2.set_xscale('log')

    #     fig1.suptitle(
    #         'Recovery errors analysis: {} - {}'.format(analysis_type, param_type),
    #         fontsize=label_size
    #     )

    #     fig2.suptitle(
    #         'Amplitudes distributions analysis: {} - {}'.format(analysis_type, param_type),
    #         fontsize=label_size
    #     )

    #     ax1_1.set_xlabel(
    #         param_type if param_name is None else param_name,
    #         fontsize=label_size
    #     )

    #     ax1_2.set_xlabel(
    #         param_type if param_name is None else param_name,
    #         fontsize=label_size
    #     )

    #     ax1_1.set_ylabel(
    #         r'$\delta_{\bf{B}}(\widehat{\bf{B}})$',
    #         fontsize=label_size
    #     )

    #     ax1_2.set_ylabel(
    #         r'$\delta_{\bf{S}}(\widehat{\bf{S}})$',
    #         fontsize=label_size
    #     )

    #     for ax in fig1.axes:
    #         ax.set_axisbelow(True)
    #         ax.grid(False, which='both')
    #         ax.grid(
    #             True,
    #             which='major',
    #             axis='both',
    #             color='gray',
    #             linestyle='--',
    #             linewidth=0.5,
    #             alpha=0.25,
    #         )

    #     ax2_1.set_axisbelow(True)
    #     ax2_1.grid(False, which='both')
    #     ax2_1.grid(
    #         True,
    #         axis='y',
    #         which='major',
    #         linestyle='--',
    #         linewidth=0.5,
    #         alpha=0.25,
    #     )

    #     n_realizations = len(parser.kept_realizations)

    #     errors = {}
    #     amplitudes = pd.DataFrame()
    #     for k, (source, source_results) in enumerate(parsed_results.items()):
    #         # Retrieve results for analysis type and parameter
    #         param_results = source_results[analysis_type][param_type]

    #         # Measure sensitivities
    #         results = __get_variations_and_errors(
    #             cfg=parser.execution_config,
    #             param_results=param_results,
    #             source=source,
    #             analysis_type=analysis_type,
    #             param_type=param_type
    #         )
    #         errors[source] = results
            

    #         # # Plot B_errors
    #         # B_errors_mean = np.mean(
    #         #     results['B_errors'],
    #         #     axis=0
    #         # )
    #         # ax1_1.plot(
    #         #     param_results['params'][0],
    #         #     B_errors_mean,
    #         #     label=source,
    #         #     color=_cmap[k]
    #         # )

    #         # # Plot s_errors
    #         # s_errors_mean = np.mean(
    #         #     results['s_errors'],
    #         #     axis=0
    #         # )
    #         # ax1_2.plot(
    #         #     param_results['params'][0],
    #         #     s_errors_mean,
    #         #     label=source,
    #         #     color=_cmap[k]
    #         # )
    #         x = np.asarray(param_results['params'][0], dtype=float)

    #         # Aproximadamente 6 posições com barras por curva
    #         step = max(1, int(np.ceil(len(x) / n_printed_stds)))

    #         for ax, key in (
    #             (ax1_1, 'B_errors'),
    #             (ax1_2, 's_errors'),
    #         ):
    #             # Formato: (realizações, pontos da varredura)
    #             values = np.asarray(results[key], dtype=float)

    #             mean = np.mean(values, axis=0)
    #             std = np.std(values, axis=0, ddof=1)/np.sqrt(n_realizations)

    #             ax.errorbar(
    #                 x,
    #                 mean,
    #                 yerr=std,
    #                 fmt='-',
    #                 color=_cmap[k],
    #                 label=source,
    #                 linewidth=1.5,
    #                 elinewidth=0.8,
    #                 capsize=3,
    #                 capthick=0.8,
    #                 errorevery=step,
    #             )

    #         # Plot coefficient amplitudes
    #         source_amplitudes = pd.DataFrame()
    #         for i, j in np.ndindex(true_B.shape):
    #             source_amplitudes = pd.concat(
    #                 [
    #                     source_amplitudes,
    #                     pd.DataFrame(
    #                         data={
    #                             'source': [source]*results['amplitudes'].shape[0],
    #                             'coefficient': ['b{}{}'.format(i+1,j+1)]*results['amplitudes'].shape[0],
    #                             'amplitude': results['amplitudes'][:,i,j]
    #                         }
    #                     )
    #                 ],
    #                 axis=0
    #             )

    #         amplitudes = pd.concat(
    #             [
    #                 amplitudes,
    #                 source_amplitudes
    #             ],
    #             axis=0
    #         ).reset_index(
    #             drop=True
    #         )

    #     ax1_1.legend(
    #         fontsize=legend_size
    #     )
    #     ax1_2.legend(
    #         fontsize=legend_size
    #     )

    #     sns.boxplot(
    #         data=amplitudes,
    #         x='source',
    #         y='amplitude',
    #         hue='coefficient',
    #         color=box_color,
    #         whis=whis_lims,
    #         width=.7,
    #         showfliers=False,
    #         ax=ax2_1
    #     )

    #     for spine in ['bottom', 'top', 'left', 'right']:
    #         ax2_1.spines[spine].set_color('black')
        
    #     ax2_1.set_xlabel(
    #         'Distribuição das fontes $r_{s}$'
    #     )
        
        
    #     # Plot vertical lines
    #     [
    #         ax2_1.axvline(
    #             x+.5,
    #             color='lightgray',
    #             linestyle='--'
    #         ) for x in ax2_1.get_xticks()
    #     ]

    #     # Save figure, if so specified
    #     if save_dir is not None:
    #         fig1.savefig(
    #             str(
    #                 (save_dir / '{}.png'.format(save_name if save_name is not None else 'recovery_errors_{}_{}'.format(analysis_type, param_type)))
    #             ),
    #             bbox_inches='tight'
    #         )
    #         fig2.savefig(
    #             str(
    #                 (save_dir / '{}.png'.format(save_name if save_name is not None else 'amplitudes_{}_{}'.format(analysis_type, param_type)))
    #             ),
    #             bbox_inches='tight'
    #         )

    #     # Standard errors of the mean across independent realizations.
    #     # Expected arrays: (realizations, parameter points).

    #     sem_tables = {}

    #     for metric in ('B_errors', 's_errors'):
    #         columns = {}

    #         for source, source_errors in errors.items():
    #             param_results = (
    #                 parsed_results[source][analysis_type][param_type]
    #             )

    #             grids = np.asarray(
    #                 param_results['params'],
    #                 dtype=float,
    #             )

    #             values = np.asarray(
    #                 source_errors[metric],
    #                 dtype=float,
    #             )

    #             # Validate the realization-by-parameter structure.
    #             if values.ndim != 2:
    #                 raise ValueError(
    #                     f'{source}/{metric}: expected a 2D array '
    #                     f'(realizations, parameter points), '
    #                     f'got shape {values.shape}.'
    #                 )

    #             if grids.shape != values.shape:
    #                 raise ValueError(
    #                     f'{source}/{metric}: parameter grids and errors '
    #                     f'must have matching shapes; got '
    #                     f'{grids.shape} and {values.shape}.'
    #                 )

    #             n_realizations, n_points = values.shape

    #             if n_realizations == 0 or n_points == 0:
    #                 raise ValueError(
    #                     f'{source}/{metric}: no observations available.'
    #                 )

    #             if not np.isfinite(grids).all():
    #                 raise ValueError(
    #                     f'{source}: non-finite parameter values found.'
    #                 )

    #             if not np.isfinite(values).all():
    #                 raise ValueError(
    #                     f'{source}/{metric}: non-finite errors found.'
    #                 )

    #             # Each parameter column must represent the same point
    #             # across all realizations.
    #             if not np.allclose(
    #                 grids,
    #                 grids[0],
    #                 rtol=1e-12,
    #                 atol=1e-14,
    #             ):
    #                 raise ValueError(
    #                     f'{source}: parameter grids differ '
    #                     'between realizations.'
    #                 )

    #             # SEM = sample standard deviation / sqrt(realization count).
    #             if n_realizations < 2:
    #                 sem = np.full(n_points, np.nan)
    #             else:
    #                 sem = (
    #                     np.std(values, axis=0, ddof=1)
    #                     / np.sqrt(n_realizations)
    #                 )

    #             index = pd.Index(
    #                 grids[0],
    #                 name=param_type,
    #             )

    #             if not index.is_unique:
    #                 raise ValueError(
    #                     f'{source}: duplicate parameter points found.'
    #                 )

    #             columns[source] = pd.Series(
    #                 sem,
    #                 index=index,
    #             )

    #         # Align sources by their actual parameter values.
    #         table = pd.DataFrame(columns).sort_index()
    #         table.columns.name = 'r_s'

    #         # Preserve the complete table for return_stats.
    #         sem_tables[metric] = table

    #         # Select a limited number of rows for printing only.
    #         if n_printed_stds is None:
    #             printed_table = table
    #         else:
    #             if (
    #                 isinstance(n_printed_stds, (bool, np.bool_))
    #                 or not isinstance(n_printed_stds, (int, np.integer))
    #                 or n_printed_stds < 1
    #             ):
    #                 raise ValueError(
    #                     'n_printed_stds must be a positive integer '
    #                     'or None to print all rows.'
    #                 )

    #             n_rows = min(n_printed_stds, len(table))

    #             row_indices = np.linspace(
    #                 0,
    #                 len(table) - 1,
    #                 num=n_rows,
    #                 dtype=int,
    #             )

    #             printed_table = table.iloc[row_indices]

    #             print(
    #                 '\n{}: standard error of the mean across realizations'.format(metric) +
    #                 f' | {analysis_type} | {param_type} | {metric}'
    #                 f' | showing {len(printed_table)}/{len(table)} points'
    #             )

    #             print(
    #                 printed_table.to_string(
    #                     float_format=lambda value: f'{value:.6e}',
    #                     na_rep='NaN',
    #                 )
    #             )

    #         if return_stats:
    #             return sem_tables


    @staticmethod
    def visualize_local_sensitivities(
        parser,
        analysis_type,
        param_type,
        true_B,
        sources=None,
        moments=('mean', 'var', 'skew', 'kurt'),
        standardize=True,
        only_converged=True,
        whis_lims=(5, 95),
        box_color='gray',
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        """Boxplots (realizações) das derivadas locais pré-calculadas pelo estimador."""
        plt.rc('legend', fontsize=legend_size, title_fontsize=legend_size)
        plt.rc('axes', labelsize=label_size)
        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)
        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {k: v for k, v in parsed_results.items() if k in sources}
        rows = []
        for source, source_results in parsed_results.items():
            pr = source_results[analysis_type][param_type]
            for r, ls, conv in zip(pr['realization'], pr['local_sensitivity'], pr['mcmc_converged']):
                if ls is None or (only_converged and conv is False):
                    continue
                for m in moments:
                    d = np.asarray(ls['derivatives'][m])
                    if standardize and m == 'mean':
                        d = d/np.sqrt(ls['moments']['var'])
                    for i, j in np.ndindex(true_B.shape):
                        rows.append({'source': source, 'realization': r, 'moment': m,
                                     'coefficient': 'b{}{}'.format(i+1, j+1), 'derivative': d[i, j]})
        data_df = pd.DataFrame(rows)
        fig, axs = plt.subplots(nrows=len(moments), ncols=1, figsize=(20, 8*len(moments)))
        axs = np.atleast_1d(axs)
        fig.suptitle('Sensibilidade local no ponto base: {} - {}'.format(analysis_type, param_type),
                     fontsize=label_size)
        for ax, m in zip(axs, moments):
            ax.axhline(0, color='lightgray', linestyle='--')
            sns.boxplot(data=data_df[data_df['moment'] == m], x='source', y='derivative',
                        hue='coefficient', palette='Greys', whis=whis_lims, width=.7,
                        showfliers=False, ax=ax)
            ax.set_ylabel(r'$\partial\,{}/\partial\lambda$'.format(m))
            ax.set_xlabel('Distribuição das fontes $r_{s}$')
            for x in ax.get_xticks():
                ax.axvline(x+.5, color='lightgray', linestyle='--')
        if save_dir is not None:
            fig.savefig(str(save_dir / '{}.png'.format(
                save_name or 'local_sensitivities_{}_{}'.format(analysis_type, param_type))),
                bbox_inches='tight')
        return data_df.groupby(['source', 'coefficient', 'moment'], as_index=False).agg(
            derivative_median=('derivative', 'median'),
            derivative_q05=('derivative', lambda v: np.quantile(v, .05)),
            derivative_q95=('derivative', lambda v: np.quantile(v, .95)),
            n_realizations=('derivative', 'count'))


    @staticmethod
    def visualize_local_mean_sensitivities(
        parser,
        analysis_type,
        param_type,
        true_B,
        sources=None,
        return_stats=False,
        moments=('mean', 'var', 'skew', 'kurt'),
        standardize=True,
        only_converged=True,
        whis_lims=(5, 95),
        box_color='gray',
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        """
            Boxplots (entre realizações) da sensibilidade local da média a posteriori
            no ponto base, d E[b_ij] / d xi, pré-calculada pelo estimador.

            Figura com 2 subplots, um abaixo do outro:
                (1) d E[b_ij] / d xi        (com sinal)
                (2) | d E[b_ij] / d xi |    (módulo, para comparar magnitudes entre coeficientes)

            Se standardize=True, a derivada é dividida pelo desvio-padrão a posteriori
            de b_ij no ponto base: (d E[b_ij] / d xi) / sqrt(Var[b_ij]).

            Assinatura idêntica a visualize_local_sensitivities; o argumento `moments`
            é mantido apenas por compatibilidade e é ignorado (só a média é usada).
        """
        # Símbolo do parâmetro varrido, para o título (xi genérico nos eixos)
        param_symbols = {
            ('prior_variation', 'mode'): r'\alpha',
            ('prior_variation', 'scale'): r'\sigma_{p}',
            ('prior_variation', 'tail_heaviness'): r'\nu_{p}',
            ('source_model_variation', 'mode'): r'\mu_{s}',
            ('source_model_variation', 'scale'): r'\sigma_{s}',
            ('source_model_variation', 'tail_heaviness'): r'\nu_{s}'
        }
        xi_symbol = param_symbols.get((analysis_type, param_type), r'\xi')
        plt.rc('legend', fontsize=legend_size, title_fontsize=legend_size)
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rc('axes', labelsize=label_size)
        plt.rc('xtick', labelsize=label_size)
        plt.rc('ytick', labelsize=label_size)
        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {k: v for k, v in parsed_results.items() if k in sources}
        # Collect mean derivatives per (source, realization, coefficient)
        rows = []
        for source, source_results in parsed_results.items():
            pr = source_results[analysis_type][param_type]
            for r, ls, conv in zip(pr['realization'], pr['local_sensitivity'], pr['mcmc_converged']):
                if ls is None or (only_converged and not bool(conv)):
                    continue
                d = np.asarray(ls['derivatives']['mean'], dtype=float)
                if standardize:
                    d = d/np.sqrt(np.asarray(ls['moments']['var'], dtype=float))
                for i, j in np.ndindex(true_B.shape):
                    rows.append({
                        'source': source,
                        'realization': r,
                        'coefficient': 'b{}{}'.format(i+1, j+1),
                        'derivative': d[i, j],
                        'abs_derivative': np.abs(d[i, j])
                    })
        data_df = pd.DataFrame(rows)
        if data_df.empty:
            raise ValueError('Nenhuma sensibilidade local disponível para {} - {}.'.format(
                analysis_type, param_type))
        source_order = [s for s in parsed_results.keys() if s in set(data_df['source'])]
        hue_order = ['b{}{}'.format(i+1, j+1) for i, j in np.ndindex(true_B.shape)]
        # Axis labels (notação dos relatórios: derivada em relação a xi)
        if standardize:
            y_signed = r'$\dfrac{\partial\,\mathbb{E}[b_{ij}]/\partial\xi}{\sqrt{\mathrm{Var}[b_{ij}]}}$'
            y_abs = r'$\dfrac{|\partial\,\mathbb{E}[b_{ij}]/\partial\xi|}{\sqrt{\mathrm{Var}[b_{ij}]}}$'
        else:
            y_signed = r'$\partial\,\mathbb{E}[b_{ij}]/\partial\xi$'
            y_abs = r'$|\partial\,\mathbb{E}[b_{ij}]/\partial\xi|$'
        # Figure: 2 subplots, one below the other
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
        fig.suptitle(
            r'Sensibilidade local da média no ponto base: {} - {} ($\xi = {}$)'.format(
                analysis_type, param_type, xi_symbol),
            fontsize=label_size
        )
        ax1.axhline(0, color='lightgray', linestyle='--')
        for ax, y, ylabel in zip(
            (ax1, ax2),
            ('derivative', 'abs_derivative'),
            (y_signed, y_abs)
        ):
            
            sns.boxplot(
                data=data_df, x='source', y=y,
                order=source_order, hue='coefficient', hue_order=hue_order,
                palette='Greys', whis=whis_lims, width=.7, showfliers=False, ax=ax
            )

            ax.set_ylabel(ylabel)

            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1.0, 1.0), title='Coeficiente', frameon=False)
            
            ax.set_xlabel('Distribuição das fontes $r_{s}$')
            
            for spine in ['bottom', 'top', 'left', 'right']:
                ax.spines[spine].set_color('black')
            
            for x in ax.get_xticks():
                ax.axvline(x+.5, color='lightgray', linestyle='--')

            ax.set_axisbelow(True)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

        ax2.set_ylim(bottom=0)
        if save_dir is not None:
            fig.savefig(str(save_dir / '{}.png'.format(
                save_name or 'local_mean_sensitivities_{}_{}'.format(analysis_type, param_type))),
                bbox_inches='tight')
        
        if return_stats:
            return data_df.groupby(['source', 'coefficient'], as_index=False, sort=False).agg(
                derivative_median=('derivative', 'median'),
                derivative_q05=('derivative', lambda v: np.quantile(v, .05)),
                derivative_q95=('derivative', lambda v: np.quantile(v, .95)),
                abs_derivative_median=('abs_derivative', 'median'),
                abs_derivative_q05=('abs_derivative', lambda v: np.quantile(v, .05)),
                abs_derivative_q95=('abs_derivative', lambda v: np.quantile(v, .95)),
                n_realizations=('derivative', 'count'))


    @staticmethod
    def IS_psis_diagnostics(
        parser,
        analysis_type,
        param_type,
        sources=None,
        params_plot=5,
        k_hat_thresh=0.7,
        return_stats=True,
        line_color='cornflowerblue',
        log_scale=False,
        save_dir=None,
        save_name=None,
        label_size=20,
        legend_size=20
    ):
        
        def __get_pareto_parameters(
            param_results
        ):
            pareto_df = pd.DataFrame()
            for r, (params, k_hat, cutoff_idx, cutoff_value, sigma) in enumerate(
                zip(
                    param_results['params'],
                    param_results['k_hat'],
                    param_results['cutoff_idx'],
                    param_results['cutoff_value'],
                    param_results['sigma']
                )
            ):
                pareto_df = pd.concat(
                    [
                        pareto_df,
                        pd.DataFrame(
                            data={
                                'params': params,
                                'k_hat': k_hat,
                                'cutoff_idx': cutoff_idx,
                                'cutoff_value': cutoff_value,
                                'sigma': sigma
                            }
                        )
                    ],
                    axis=0
                )

            # Quantile functions
            quantile_975_fn = lambda x: np.quantile(x, 0.975)
            quantile_025_fn = lambda x: np.quantile(x, 0.025)

            return pareto_df.groupby(
                by='params',
                as_index=False
            ).agg(
                k_hat_median=('k_hat', 'median'),
                k_hat_975=('k_hat', quantile_975_fn),
                k_hat_025=('k_hat', quantile_025_fn),
                cutoff_idx_median=('cutoff_idx', 'median'),
                cutoff_idx_975=('cutoff_idx', quantile_975_fn),
                cutoff_idx_025=('cutoff_idx', quantile_025_fn),
                cutoff_value_median=('cutoff_value', 'median'),
                cutoff_value_975=('cutoff_value', quantile_975_fn),
                cutoff_value_025=('cutoff_value', quantile_025_fn),
                sigma_median=('sigma', 'median'),
                sigma_975=('sigma', quantile_975_fn),
                sigma_025=('sigma', quantile_025_fn),
            )
        

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

        parsed_results = parser.parsed_results

        if sources is not None:
            parsed_results = {
                k: v for k, v in parsed_results.items() if k in sources
            }

        # Iterate in sources
        plot_df = pd.DataFrame()
        for source, source_results in parsed_results.items():

            # Retrieve results for analysis type and parameter
            param_results = source_results[analysis_type][param_type]
            
            # Get pareto parameters for different realizations
            pareto_df = __get_pareto_parameters(
                param_results=param_results
            )

            # Create plot
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3, ncols=1,
                figsize=(20,25)
            )

            fig.suptitle(
                'Pareto Parameters: {} - {} - {}'.format(
                    source,
                    analysis_type,
                    param_type
                ),
                fontsize=label_size
            )

            if log_scale:
                ax1.set_xscale('log')
                ax2.set_xscale('log')
                ax3.set_xscale('log')
            # Pareto k-hat horizontal lines at 0 and thresh
            [
                ax1.axhline(
                    v,
                    linestyle='--',
                    color='gray'
                ) for v in [0, k_hat_thresh]
            ]
            # Median lines
            [
                ax.plot(
                    pareto_df['params'].values,
                    pareto_df['{}_median'.format(p)],
                    color=line_color,
                    linewidth=5
                ) for ax, p in zip(
                    [ax1, ax2, ax3],
                    ['k_hat', 'cutoff_value', 'sigma']
                )
            ]

            # 2.5% lines
            [
                ax.plot(
                    pareto_df['params'].values,
                    pareto_df['{}_025'.format(p)],
                    color=line_color,
                    linestyle='--'
                ) for ax, p in zip(
                    [ax1, ax2, ax3],
                    ['k_hat', 'cutoff_value', 'sigma']
                )
            ]

            # 97.5% lines
            [
                ax.plot(
                    pareto_df['params'].values,
                    pareto_df['{}_975'.format(p)],
                    color=line_color,
                    linestyle='--'
                ) for ax, p in zip(
                    [ax1, ax2, ax3],
                    ['k_hat', 'cutoff_value', 'sigma']
                )
            ]

            # Fill
            [
                ax.fill_between(
                    pareto_df['params'].values,
                    pareto_df['{}_025'.format(p)],
                    pareto_df['{}_975'.format(p)],
                    color=line_color,
                    alpha=0.2
                ) for ax, p in zip(
                    [ax1, ax2, ax3],
                    ['k_hat', 'cutoff_value', 'sigma']
                )
            ]

            [
                ax.set_xlabel(
                    param_type,
                    fontsize=label_size
                ) for ax in [ax1, ax2, ax3]
            ]

            [
                ax.set_ylabel(
                    p,
                    fontsize=label_size
                ) for ax, p in zip(
                    [ax1, ax2, ax3],
                    [r'$\hat{k}$', 'u', r'$\sigma$']
                )
            ]

            for ax in (
                ax1,
                ax2,
                ax3
            ):
                ax.set_axisbelow(True)
                ax.grid(
                    True,
                    which='major',
                    color='lightgray',
                    linestyle='--',
                    linewidth=0.5
                )

            # Save figure, if so specified
            if save_dir is not None:
                fig.savefig(
                    str(
                        (save_dir / '{}_.png'.format(save_name if save_name is not None else 'psis_diagnostics_{}_{}_{}'.format(analysis_type, param_type, source)))
                    ),
                    bbox_inches='tight'
                )
                
        # Calculate statistics, if so specified
        if return_stats:
            return pareto_df
        

    @staticmethod
    def summarize_sensitivities(
        parser,
        sources=None,
        analyses=None,
        baseline_values=None,
        only_converged=True,
        dominance_ratio=2.0,
        float_format='{:.3f}',
        return_stats=True
    ):
        """
            Prints a summary of prior vs. likelihood (source model) sensitivities of the posterior mean.

            Scores (computed per realization, then the median across realizations is reported):
                sweep score      : max_{xi in grid} max_ij |E_xi[b_ij] - E_0[b_ij]| / sd_0(b_ij)
                derivative score : max_ij |dE[b_ij]/dxi| / sd_0(b_ij), evaluated at the base point

            Args:
                parser: parser object with a parsed_results attribute.
                sources: optional list of sources (r_s) to include.
                analyses: optional list of (analysis_type, param_type) pairs to include.
                baseline_values: {(analysis_type, param_type): xi_0 in grid units}, used by the
                    sweep score. If omitted, the base point is the grid point whose raw IS
                    log-weights have the smallest spread (all weights are equal at xi_0).
                only_converged: if True, uses only realizations flagged as converged.
                dominance_ratio: likelihood/prior ratio above which (or below whose inverse)
                    one component is said to dominate.
                float_format: format used for printed numbers.
                return_stats: if True, returns a dict of DataFrames.
        """
        baseline_values = baseline_values or {}
        component_names = {
            'prior_variation': 'prior',
            'source_model_variation': 'likelihood'
        }

        def __base_index(pr, analysis_type, param_type, grid):
            key = (analysis_type, param_type)
            if key in baseline_values:
                return int(np.argmin(np.abs(grid - baseline_values[key])))
            spread = np.median(
                np.stack([
                    [
                        np.std(np.log(np.maximum(np.asarray(w, dtype=float), np.finfo(float).tiny)))
                        for w in realization_weights
                    ] for realization_weights in pr['raw_weights']
                ]),
                axis=0
            )
            return int(np.argmin(spread))

        parsed_results = parser.parsed_results
        if sources is not None:
            parsed_results = {k: v for k, v in parsed_results.items() if k in sources}

        rows = []
        for source, source_results in parsed_results.items():
            for analysis_type, analysis_results in source_results.items():
                if analysis_type not in component_names or len(analysis_results) == 0:
                    continue
                for param_type, pr in analysis_results.items():
                    if analyses is not None and (analysis_type, param_type) not in analyses:
                        continue

                    grids = np.asarray(pr['params'], dtype=float)          # (R, P)
                    means = np.asarray(pr['means'], dtype=float)           # (R, P, 2, 2)
                    variances = np.asarray(pr['vars'], dtype=float)        # (R, P, 2, 2)
                    n_real = grids.shape[0]

                    if not np.allclose(grids, grids[0]):
                        raise ValueError(f'{source}/{analysis_type}/{param_type}: grids differ between realizations.')
                    if means.shape[:2] != grids.shape or variances.shape != means.shape:
                        raise ValueError(f'{source}/{analysis_type}/{param_type}: inconsistent array shapes.')

                    local = pr.get('local_sensitivity', [None] * n_real)
                    converged = pr.get('mcmc_converged', [True] * n_real)
                    if len(local) != n_real or len(converged) != n_real:
                        raise ValueError(f'{source}/{analysis_type}/{param_type}: per-realization lists have different lengths.')

                    i0 = __base_index(pr, analysis_type, param_type, grids[0])

                    for r in range(n_real):
                        if only_converged and not bool(converged[r]):
                            continue

                        # Sweep score
                        sd0 = np.sqrt(variances[r, i0])
                        sweep_score = np.max(np.abs(means[r] - means[r, i0]) / sd0)

                        # Derivative score
                        derivative_score = np.nan
                        ls = local[r]
                        if ls is not None:
                            d = np.asarray(ls['derivatives']['mean'], dtype=float)
                            sd_ls = np.sqrt(np.asarray(ls['moments']['var'], dtype=float))
                            derivative_score = np.max(np.abs(d) / sd_ls)

                        rows.append({
                            'component': component_names[analysis_type],
                            'param_type': param_type,
                            'source': source,
                            'realization': r,
                            'sweep_score': sweep_score,
                            'derivative_score': derivative_score
                        })

        data_df = pd.DataFrame(rows)
        if data_df.empty:
            raise ValueError('No sensitivity results found for the requested selection.')

        stats_df = data_df.groupby(
            ['component', 'param_type', 'source'], as_index=False, sort=False
        ).agg(
            n_realizations=('realization', 'count'),
            sweep_score=('sweep_score', 'median'),
            derivative_score=('derivative_score', 'median')
        )

        # Prior vs. likelihood: same parameter role, derivative score
        comparison = []
        for source, df in stats_df.groupby('source', sort=False):
            for param_type in df['param_type'].unique():
                pri = df[(df['component'] == 'prior') & (df['param_type'] == param_type)]
                lik = df[(df['component'] == 'likelihood') & (df['param_type'] == param_type)]
                if pri.empty or lik.empty:
                    continue
                p = float(pri['derivative_score'].iloc[0])
                l = float(lik['derivative_score'].iloc[0])
                ratio = l / p if p > 0 else np.inf
                if not (np.isfinite(p) and np.isfinite(l)):
                    verdict = 'incomplete'
                elif ratio >= dominance_ratio:
                    verdict = 'likelihood dominates'
                elif ratio <= 1 / dominance_ratio:
                    verdict = 'prior dominates'
                else:
                    verdict = 'comparable'
                comparison.append({
                    'source': source,
                    'param_type': param_type,
                    'prior': p,
                    'likelihood': l,
                    'likelihood/prior': ratio,
                    'verdict': verdict
                })
        comparison_df = pd.DataFrame(comparison)

        # ---------------- Printing ----------------
        fmt = lambda v: float_format.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
        line = '=' * 100

        print(line)
        print('SENSITIVITY SUMMARY - posterior mean of b_ij (medians across realizations)')
        print(f'Converged realizations only: {only_converged}')
        print()
        print('Sweep score      = max over grid and coefficients of |E_xi[b_ij] - E_0[b_ij]| / sd_0(b_ij)')
        print('                   -> largest mean displacement over the swept range, in base posterior sds.')
        print('                      Depends on the chosen sweep range.')
        print('Derivative score = max over coefficients of |dE[b_ij]/dxi| / sd_0(b_ij), at the base point')
        print('                   -> base posterior sds moved per unit change of xi. Range-free, but in')
        print('                      units of 1/xi: compare only parameters with the same units.')
        print(line)

        for metric, title in (
            ('sweep_score', '(1) Sweep score'),
            ('derivative_score', '(2) Derivative score')
        ):
            table = stats_df.pivot_table(
                index=['component', 'param_type'], columns='source', values=metric, sort=False
            )
            table.columns.name = 'r_s'
            print('\n' + title)
            print(table.to_string(float_format=fmt))

        if not comparison_df.empty:
            print(f'\n(3) Prior vs. likelihood (source model), derivative score')
            print(f'    verdict: likelihood/prior >= {dominance_ratio} -> likelihood dominates; '
                  f'<= 1/{dominance_ratio} -> prior dominates; otherwise comparable')
            for param_type, df in comparison_df.groupby('param_type', sort=False):
                print(f'\n  {param_type}')
                print(df.drop(columns='param_type').to_string(index=False, float_format=fmt))
        print(line)

        if return_stats:
            return {
                'realizations': data_df,
                'summary': stats_df,
                'comparison': comparison_df
            }

     

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


        

        
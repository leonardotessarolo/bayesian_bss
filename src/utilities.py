import pandas as pd
import numpy as np

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

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
            '$B_{00}$',
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
            '$B_{01}$',
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
            '$B_{10}$',
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
            '$B_{11}$',
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
        save_dir,
        ylims=None,
        xlims=None,
        label_size=15,
        tick_size=15,
        legend_size=15,
        plot_estimate=True
    ):

        print('-'*100)
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
        axs[0,0].hist(
            valid_samples[:, 0, 0],
            density=True,
            bins=nbins,
            label='samples'
        )
        if plot_estimate:
            axs[0,0].axvline(
                B_est[0, 0],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label='$\hat{B}_{00}$'
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
            label='samples'
        )
        if plot_estimate:
            axs[0,1].axvline(
                B_est[0, 1],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label='$\hat{B}_{01}$'
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
            label='samples'
        )
        if plot_estimate:
            axs[1,0].axvline(
                B_est[1, 0],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label='$\hat{B}_{10}$'
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
            label='samples'
        )
        if plot_estimate:
            axs[1,1].axvline(
                B_est[1, 1],
                color='limegreen',
                linestyle='--',
                linewidth=4,
                label='$\hat{B}_{11}$'
            )
        axs[1,1].set_xlabel(
            '$B_{11}$',
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
        sns.lineplot(
            data=logs,
            x='iteration',
            y='log_posterior'
        )
        plt.ylabel(
            '$\log P(\hat{B}|X)$',
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
            label='$\hat{s}_{0}$',
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
            '$\hat{s}_{0}$',
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
            label='$\hat{s}_{1}$',
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
            '$\hat{s}_{1}$',
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
            '$\log P(\hat{B}|X)$',
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
            '$\det(\hat{B})$',
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
            '$d \;\log P(\hat{B}|X)/d \; bij$',
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
            label='$\hat{s}_{0}$',
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
            '$\hat{s}_{0}$',
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
            label='$\hat{s}_{1}$',
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
            '$\hat{s}_{1}$',
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
        label_size=15,
        tick_size=15,
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
        
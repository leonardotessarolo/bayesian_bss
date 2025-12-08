import pandas as pd
import numpy as np

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

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
                bin_cfg='auto'
            )
            ax2.hist(
                x=s[i-1,:],
                bins=100,
                cumulative=True,
                density=True,
                label='s{}'.format(i-1),
                alpha=0.5,
                bin_cfg='auto'
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
            figsize=(25,9)
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
                label='$\hat{b}_{11}$'
            )
            
        axs[0,0].set_xlabel(
            '$b_{11}$',
            fontsize=label_size
        )
        axs[0,0].set_ylabel(
            'density',
            fontsize=label_size
        )
        # axs[0,0].set_title(
        #     '(a)',
        #     fontsize=label_size
        # )
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
                label='$\hat{b}_{12}$'
            )
        axs[0,1].set_xlabel(
            '$b_{12}$',
            fontsize=label_size
        )
        axs[0,1].set_ylabel(
            'density',
            fontsize=label_size
        )
        # axs[0,1].set_title(
        #     '(b)',
        #     fontsize=label_size
        # )
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
                label='$\hat{b}_{21}$'
            )
        axs[1,0].set_xlabel(
            '$b_{21}$',
            fontsize=label_size
        )
        axs[1,0].set_ylabel(
            'density',
            fontsize=label_size
        )
        # axs[1,0].set_title(
        #     '(c)',
        #     fontsize=label_size
        # )
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
                label='$\hat{b}_{22}$'
            )
        axs[1,1].set_xlabel(
            '$b_{22}$',
            fontsize=label_size
        )
        axs[1,1].set_ylabel(
            'density',
            fontsize=label_size
        )
        # axs[1,1].set_title(
        #     '(d)',
        #     fontsize=label_size
        # )
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
        save_dir=None,
        label_size=15,
        tick_size=15,
    ):

        print('-'*100)
        print('Normalized log-posterior (average)')
        print('-'*100)
        
        fig = plt.figure(figsize=(20,7))

        U, V = np.meshgrid(u_vec, v_vec)
        
        plt.contour(
            U,
            V,
            z.T,
            levels=50,
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

        plt.xticks(
            fontsize=tick_size
        )
        plt.yticks(
            fontsize=tick_size
        )
        plt.legend(
            loc='upper right',
            fontsize=15
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
    
        # Create plot
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(20,7)
        )
        if equal_aspect:
            ax.set_aspect('equal')
        sns.scatterplot(
            data=plot_df,
            x='u',
            y='v',
            hue='test_case',
            style='test_case',
            # markers = ['+', 'x'],
            markers = ['s', '>', '<','o'],
            **{
                's': marker_size
            }
        )
        plt.title(
            'Casos de teste: {}'.format(', '.join(test_cases)),
            fontsize=15
        )
        plt.xlim(u_range)
        plt.ylim(v_range)

        # Calculate statistics, if so specified
        if return_stats:
            stats_df = plot_df.groupby(
                by='test_case',
                as_index=False
            ).agg(
                u_mean=('u', 'mean'),
                v_mean=('v', 'mean'),
                u_std=('u', 'std'),
                v_std=('v', 'std')
            )
            return stats_df


    @classmethod
    def plot_test_cases_mmse_coefficient_estimates(
        cls,
        parser,
        test_cases=None,
        B_true=None,
        bin_cfg='auto',
        return_stats=True
        # u_range=(-0.3, 0.3),
        # v_range=(-0.3, 0.3),

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

        fig.suptitle(
            'Casos de teste: {}'.format(', '.join(test_cases)),
            fontsize=25
        )

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
            data=plot_df,
            x='b11',
            hue='test_case',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[0,0]
        )

        # B12
        sns.histplot(
            data=plot_df,
            x='b12',
            hue='test_case',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[0,1]
        )

        # B21
        sns.histplot(
            data=plot_df,
            x='b21',
            hue='test_case',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[1,0]
        )

        # B22
        sns.histplot(
            data=plot_df,
            x='b22',
            hue='test_case',
            hue_order=test_cases,
            bins=bin_cfg,
            ax=axs[1,1]
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
        fig = plt.figure(figsize=(20,7))
        sns.histplot(
            data=plot_df,
            x='errors',
            hue='test_case',
            bins=bin_cfg
        )
        plt.title(
            'Casos de teste: {}'.format(', '.join(test_cases)),
            fontsize=15
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


        

        
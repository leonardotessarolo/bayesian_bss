import pandas as pd
import numpy as np

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
        save_dir
    ):
        # Plot sampled coefficients
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        fig.suptitle(
            'Evolution of sampled coefficients',
            fontsize=25
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
            fontsize=15
        )
        axs[0,0].set_ylabel(
            '$B_{00}$',
            fontsize=15
        )
        axs[0,0].legend(
            fontsize=15
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
            fontsize=15
        )
        axs[0,1].set_ylabel(
            '$B_{01}$',
            fontsize=15
        )
        axs[0,1].legend(
            fontsize=15
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
            fontsize=15
        )
        axs[1,0].set_ylabel(
            '$B_{10}$',
            fontsize=15
        )
        axs[1,0].legend(
            fontsize=15
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
            fontsize=15
        )
        axs[1,1].set_ylabel(
            '$B_{11}$',
            fontsize=15
        )
        axs[1,1].legend(
            fontsize=15
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
        save_dir
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
                    'interval': ['[{},{}['.format(start, end)]*(end-start)
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
        
        # Plot coefficient distribution evolution
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        fig.suptitle(
            'Evolution of coefficient distributions',
            fontsize=25
        )
        
        # B_00
        sns.kdeplot(
            data=plot_df,
            x='B_00',
            hue='interval',
            ax=axs[0,0],
            palette=palette
        )
        
        # B_01
        sns.kdeplot(
            data=plot_df,
            x='B_01',
            hue='interval',
            ax=axs[0,1],
            palette=palette
        )
        
        # B_10
        sns.kdeplot(
            data=plot_df,
            x='B_10',
            hue='interval',
            ax=axs[1,0],
            palette=palette
        )
        
        # B_11
        sns.kdeplot(
            data=plot_df,
            x='B_11',
            hue='interval',
            ax=axs[1,1],
            palette=palette
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
        save_dir
    ):
        # Plot sampled coefficients
        fig, axs = plt.subplots(
            nrows=2, ncols=2,
            figsize=(25,15)
        )
        
        fig.suptitle(
            'Marginal distributions of sampled coefficients (post-burn-in)',
            fontsize=25
        )
        
        # B_00
        axs[0,0].hist(
            valid_samples[:, 0, 0],
            density=True,
            bins=nbins,
            label='samples'
        )
        axs[0,0].axvline(
            B_est[0, 0],
            color='limegreen',
            linestyle='--',
            linewidth=4,
            label='$\hat{B}_{00}$'
        )
        axs[0,0].set_xlabel(
            '$B_{00}$',
            fontsize=15
        )
        axs[0,0].set_ylabel(
            'density',
            fontsize=15
        )
        axs[0,0].legend(
            loc='upper right',
            fontsize=15
        )
        
        
        # B_01
        axs[0,1].hist(
            valid_samples[:, 0, 1],
            density=True,
            bins=nbins,
            label='samples'
        )
        axs[0,1].axvline(
            B_est[0, 1],
            color='limegreen',
            linestyle='--',
            linewidth=4,
            label='$\hat{B}_{01}$'
        )
        axs[0,1].set_xlabel(
            '$B_{01}$',
            fontsize=15
        )
        axs[0,1].set_ylabel(
            'density',
            fontsize=15
        )
        axs[0,1].legend(
            loc='upper right',
            fontsize=15
        )
        
        # B_10
        axs[1,0].hist(
            valid_samples[:, 1, 0],
            density=True,
            bins=nbins,
            label='samples'
        )
        axs[1,0].axvline(
            B_est[1, 0],
            color='limegreen',
            linestyle='--',
            linewidth=4,
            label='$\hat{B}_{10}$'
        )
        axs[1,0].set_xlabel(
            '$B_{10}$',
            fontsize=15
        )
        axs[1,0].set_ylabel(
            'density',
            fontsize=15
        )
        axs[1,0].legend(
            loc='upper right',
            fontsize=15
        )
        
        # B_11
        axs[1,1].hist(
            valid_samples[:, 1, 1],
            density=True,
            bins=nbins,
            label='samples'
        )
        axs[1,1].axvline(
            B_est[1, 1],
            color='limegreen',
            linestyle='--',
            linewidth=4,
            label='$\hat{B}_{11}$'
        )
        axs[1,1].set_xlabel(
            '$B_{11}$',
            fontsize=15
        )
        axs[1,1].set_ylabel(
            'density',
            fontsize=15
        )
        axs[1,1].legend(
            loc='upper right',
            fontsize=15
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
        save_dir
    ):
        fig = plt.figure(figsize=(20,7))
        sns.lineplot(
            data=logs,
            x='iteration',
            y='log_posterior'
        )
        plt.ylabel(
            '$\log P(B|X)$'
        )
        t = plt.title(
            'Evolution of log-posterior',
            fontsize=20
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
        save_dir
    ):
        
        
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
            fontsize=15
        )
        axs[0,0].set_title(
            'Time series - true values and estimates - coefficient s0',
            fontsize=15
        )
        axs[0,0].legend(fontsize=15)
        
        # Axs 01
        axs[0,1].scatter(
            s[0,:],
            s_est[0,:],
        )
        axs[0,1].set_xlabel(
            '$s_{0}$',
            fontsize=15
        )
        axs[0,1].set_ylabel(
            '$\hat{s}_{0}$',
            fontsize=15
        )
        axs[0,1].set_title(
            'Scatter plot - true values and estimates - coefficient s0',
            fontsize=15
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
            fontsize=15
        )
        axs[1,0].set_title(
            'Time series - true values and estimates - coefficient s1',
            fontsize=15
        )
        axs[1,0].legend(fontsize=15)
        
        # Axs 11
        axs[1,1].scatter(
            s[1,:],
            s_est[1,:],
        )
        axs[1,1].set_xlabel(
            '$s_{1}$',
            fontsize=15
        )
        axs[1,1].set_ylabel(
            '$\hat{s}_{1}$',
            fontsize=15
        )
        axs[1,1].set_title(
            'Scatter plot - true values and estimates - coefficient s1',
            fontsize=15
        )

        if save_dir is not None:
            fig.savefig(
                str(
                    (save_dir / 'source_separation_results.png')
                )
            )

    
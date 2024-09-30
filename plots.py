import pandas as pd
import matplotlib.pyplot as plt

# General function to load data from multiple CSVs and plot the graphs
def plot_from_multiple_csvs(file_paths, labels, save_path='plot.png'):
    # Ensure that the number of file paths and labels match
    if len(file_paths) != len(labels):
        raise ValueError("The number of file paths must match the number of labels.")
    
    # Prepare lists to store data for discrete and continuous factors
    discrete_dfs = []
    continuous_dfs = []
    
    # Loop through each CSV file and process the data
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        
        # Combine 'modality' and 'factor_name' to ensure uniqueness
        df['unique_factor'] = df['modality'] + '_' + df['factor_name']
        
        # Separate the data into discrete and continuous factors
        df_discrete = df[df['factor_type'] == 'discrete'].dropna(subset=['acc_logreg', 'acc_mlp'])
        df_continuous = df[df['factor_type'] == 'continuous'].dropna(subset=['r2_linreg', 'r2_krreg'])
        
        # Append to lists
        discrete_dfs.append(df_discrete)
        continuous_dfs.append(df_continuous)
    
    # Set up two separate plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First plot: Discrete factors (Accuracy)
    ax1.set_title('Classification Accuracy for Discrete Factors')
    ax1.set_xlabel('Factor Name (Modality_FactorName)')
    ax1.set_ylabel('Accuracy')

    # Loop through each dataframe for discrete data and plot it
    for i, df_discrete in enumerate(discrete_dfs):
        label = labels[i]
        ax1.plot(df_discrete['unique_factor'], df_discrete['acc_logreg'], marker='o', label=f'LogReg ({label})', linestyle='--')
        ax1.plot(df_discrete['unique_factor'], df_discrete['acc_mlp'], marker='x', label=f'MLP ({label})', linestyle='-')
    
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', rotation=45)

    # Second plot: Continuous factors (R² values)
    ax2.set_title('R² Values for Continuous Factors')
    ax2.set_xlabel('Factor Name (Modality_FactorName)')
    ax2.set_ylabel('R² Value')

    # Loop through each dataframe for continuous data and plot it
    for i, df_continuous in enumerate(continuous_dfs):
        label = labels[i]
        ax2.plot(df_continuous['unique_factor'], df_continuous['r2_linreg'], marker='o', label=f'LinReg ({label})', linestyle='--')
        ax2.plot(df_continuous['unique_factor'], df_continuous['r2_krreg'], marker='x', label=f'KRReg ({label})', linestyle='-')
    
    ax2.legend(loc='upper left')
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save the plot to the specified file path
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

# Example usage (replace with the paths to your CSV files and corresponding labels)
# plot_from_multiple_csvs(['encoding_4.csv', 'encoding_8.csv'], ['Encoding 4', 'Encoding 8'], save_path='comparison_plot.png')



# Example usage (replace with the paths to your CSV files and corresponding labels)
plot_from_multiple_csvs(['models/imgtxt_exp_run_encoding_size_04/results.csv', 'models/imgtxt_exp_run_encoding_size_08/results.csv'], ['Encoding_Size_4', 'Encoding_Size_8'])

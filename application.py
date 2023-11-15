from fredapi import Fred
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tsa.stattools import adfuller


''' THIS CODE WAS ASSEMBLED QUICKYL AND HAS NOT BEEN REFACTORED. THERE'S A GOOD BIT OF 
    TECHINICAL DEBT'''

def get_data():
    api_key = 'YOUR_KEY_HERE'
    fred = Fred(api_key=api_key)

    oil = 'DCOILWTICO'
    jet_fuel = 'WJFUELUSGULF'

    # Retrieve series
    oil_series = fred.get_series(oil)
    jet_fuel_series = fred.get_series(jet_fuel)

    # Convert to DataFrames
    oil_df = pd.DataFrame(oil_series, columns=['OilPrice'])
    jet_fuel_df = pd.DataFrame(jet_fuel_series, columns=['JetFuel'])

    # Merge DataFrames
    combined_df = pd.merge(oil_df, jet_fuel_df, left_index=True, right_index=True, how='outer')
    combined_df = combined_df.dropna()
    df = combined_df.copy()
    combined_df['OilPrice'] = combined_df['OilPrice'].pct_change()
    combined_df['JetFuel'] = combined_df['JetFuel'].pct_change()

    return combined_df.dropna(), df


def line_grapher(df):
    print(df)
    if df.shape[1] != 2:
        raise ValueError("DataFrame should have exactly two columns")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_first = 'tab:red'
    ax1.set_xlabel('Date', fontsize=14, weight='bold')
    ax1.set_ylabel("WTI Spot PRice", color=color_first, fontsize=14, weight='bold')
    ax1.plot(df.index, df.iloc[:, 0], color=color_first, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color_first, labelsize=12)

    ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')

    ax2 = ax1.twinx()
    color_second = 'blue'
    ax2.set_ylabel('Jet Fuel Spot Price', color=color_second, fontsize=14, weight='bold')
    ax2.plot(df.index, df.iloc[:, 1], color=color_second, linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=color_second, labelsize=12)

    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(fontsize=12, rotation=45)
    fig.autofmt_xdate()

    plt.title('WTI & Jet Fuel Price', fontsize=16, weight='bold')
    legends = [df.columns[0], 'Jet Fuel Price']  # Updated legend
    colors = [color_first, color_second]
    custom_legends = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    ax1.legend(custom_legends, legends, fontsize=12, loc='upper left')

    plt.figtext(0.15, 0.02, 'Data Sources: FRED API | DCOILWTICO & WJFUELUSGULF', ha='center', va='center', fontsize=8, color='grey')

    plt.tight_layout()
    sns.despine()

    plt.savefig('AAL_spotfuel_fuelexp.png', dpi=300)

    # plt.show()


def graph_regression(df):

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.regplot(x='OilPrice', y='JetFuel', data=df,
                scatter_kws={'color': 'darkgreen', 'alpha': 0.6, 's': 50},  # scatter style
                line_kws={'color': 'navy', 'ls': '--', 'lw': 2})  # line style

    slope, intercept = np.polyfit(df['OilPrice'], df['JetFuel'], 1)
    equation = f"b1 = {slope:.4f}x + {intercept:.4f}"
    correlation_matrix = np.corrcoef(df['OilPrice'], df['JetFuel'])
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    plt.text(0.05, 0.85, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.title('WTI X Jet Fuel Price Regression Line', fontsize=16, fontweight='bold')
    plt.xlabel('Oil Price', fontsize=14)
    plt.ylabel('Jet Fuel Price', fontsize=14)

    plt.savefig('correlation_plot.png', dpi=300)
    plt.show()


def regression_details(df):
    if len(df.columns) != 2:
        raise ValueError("DataFrame must contain exactly two columns of data.")

    # Assuming the first column is FuelPrice and the second is FuelExpense.
    X = df.iloc[:, 0]  # Independent variable (predictor)
    y = df.iloc[:, 1]  # Dependent variable (outcome)

    X = sm.add_constant(X)  # Add an intercept to the model

    model = sm.OLS(y, X).fit()  # Fit the OLS model

    # Get the summary of the model
    summary = model.summary()

    # Calculate the Standard Error of the Estimate (SEE)
    see = np.sqrt(model.mse_resid)

    r_value, _ = pearsonr(df.iloc[:, 0], df.iloc[:, 1])

    # Print the summary and the SEE
    print(summary)
    print(f"Standard Error of the Estimate (SEE): {see}")
    print(f"Pearson correlation coefficient (r-value): {r_value}")
    print()
    print("==="*30)

    model = ols('JetFuel ~ OilPrice', data=df).fit()
    anova_results = anova_lm(model, typ=2)

    print("ANOVA Results")
    print(anova_results)

    return model


def t_test(model, df, alpha=0.025):
    b1 = model.params[1]
    std_err = model.bse[1]
    df_resid = model.df_resid

    t_statistic = b1 / std_err

    t_critical = stats.t.ppf(1 - alpha / 2, df_resid)

    if isinstance(t_critical, np.ndarray):
        if t_critical.size == 1:
            t_critical = t_critical.item()  # Convert to scalar
        else:
            raise ValueError("Critical t value calculation is returning an array instead of a single scalar.")

    rejection_region = f"t > {t_critical:.3f} or t < {-t_critical:.3f}"
    reject_null = abs(t_statistic) > t_critical

    result = {
        'H0': 'b1=0',
        'Ha': 'b1≠0',
        'b1': b1,
        's (b1)': std_err,
        't': t_statistic,
        'DF': df,
        'Level of Significance': alpha,
        'Critical Value t(0.025, df)': t_critical,
        'Rejection Region': rejection_region,
        'Conclusion': 'Reject the null hypothesis' if reject_null else 'Do not reject the null hypothesis'
    }

    print("===" *30)
    print()
    print(" " * 10, "T-Test")
    for key, value in result.items():
        print(f"{key}: {value}")
    print()
    print("===" * 30)



    return result


def f_test(df, model):

    # RSS - Residual Sum of Squares
    rss = sum(model.resid ** 2)

    # SSE
    sse = sum((model.fittedvalues - df.iloc[:, 1].mean()) ** 2)

    # Degrees of freedom
    df_regression = model.df_model  # Number of explanatory variables in the model
    df_residual = model.df_resid  # Degrees of freedom of residuals

    # F-statistic
    f_stat = (sse / df_regression) / (rss / df_residual)

    # F critical value for a given alpha (e.g., alpha = 0.05 for a 95% confidence level)
    alpha = 0.05
    f_critical = stats.f.ppf(1 - alpha, df_regression, df_residual)

    # Output results
    print("==="*30)
    print("F-Test")
    print()
    print(f"H0: b1=0")
    print(f"Ha: b1≠0")
    print(f"RSS: {rss}")
    print(f"SSE: {sse}")
    print(f"N-2: {df_residual}")
    print(f"F: {f_stat}")
    print(f"F({df_regression},{df_residual}): {f_critical}")

    # Check if we should reject the null hypothesis
    if f_stat > f_critical:
        print(
            f"Since the F-value is higher than {f_critical:.2f}, we reject the null hypothesis with high significance.")
    else:
        print(f"Since the F-value is lower than {f_critical:.2f}, we fail to reject the null hypothesis.")

    print("==="*30)
    return {
        'RSS': rss,
        'SSE': sse,
        'DF Regression': df_regression,
        'DF Residual': df_residual,
        'F Statistic': f_stat,
        'F Critical': f_critical,
        'Reject Null': f_stat > f_critical
    }


def mean_revision(df):
    adf_test = adfuller(df.JetFuel)

    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])

    df['Price_Diff'] = df.JetFuel.diff()

    # Create lagged price
    df['Lagged_Price'] = df['Price_Diff'].shift()

    # Remove NaN values
    data = df.dropna()

    # Define the dependent and independent variables
    X = data['Lagged_Price']
    y = data['Price_Diff']

    # Add a constant to the independent variable
    X = sm.add_constant(X)

    # Fit the AR(1) model
    model = sm.OLS(y, X).fit()

    b0, b1 = model.params

    # Calculate mean reverting level (xt)
    xt = b0 / (1 - b1)

    # Calculate AR(1)
    ar_1 = b0 + b1 * xt

    # Print results
    print(f"Mean Reverting Level (xt): {xt}")
    print(f"AR(1): {ar_1}")

    current_price = df.JetFuel.iloc[-1]  # Get the most recent price
    periods = 0
    projected_price = current_price
    threshold = .01
    max_periods = 100

    # Iterate until the projected price is close to the mean reverting level
    while abs(projected_price - xt) > threshold and periods < max_periods:
        projected_price = b0 + b1 * projected_price
        periods += 1

    print(f"Estimated Periods for Mean Reversion: {periods}")


def excel_outputs(df):
    X = sm.add_constant(df['OilPrice'])  # Adds a constant term to the predictor
    model = sm.OLS(df['JetFuel'], X).fit()

    # Calculate residuals
    df['Residuals'] = model.resid

    # Calculate squared residuals
    df['Squared_Residuals'] = df['Residuals'] ** 2

    # Calculate deviation from the mean
    mean_Y = df['JetFuel'].mean()
    df['Deviation_From_Mean'] = df['JetFuel'] - mean_Y

    # Calculate squared deviations
    df['Squared_Deviations'] = df['Deviation_From_Mean'] ** 2

    df.to_csv('excel_output.csv')


if __name__ == "__main__":
    df, dx = get_data()
    line_grapher(dx)
    graph_regression(df)
    model = regression_details(df)
    t_test(model, df)
    f_test(df, model)
    mean_revision(dx)
    excel_outputs(df)


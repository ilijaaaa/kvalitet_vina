import seaborn as sb
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor    
sb.set(font_scale=1.)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from utils_nans1 import *

def are_assumptions_satisfied(model, x, y, p_value_thresh=0.05, plt=False):
    '''provera pretpostavki'''
    x_with_const = sm.add_constant(x)
    # Linearity
    is_linearity_found, p_value = linear_assumption(model, x_with_const, y, p_value_thresh, plt)
    if(plt):
        print('Linearity:')
        print(is_linearity_found, p_value)
    # Independence of errors
    autocorrelation, dw_value = independence_of_errors_assumption(model, x_with_const, y, plt)
    if(plt):
        print('Independence of errors:')
        print(autocorrelation, dw_value)
    # Normality of errors
    n_dist_type, p_value = normality_of_errors_assumption(model, x_with_const, y, p_value_thresh, plt)
    if(plt):
        print('Normality of errors:')
        print(n_dist_type, p_value)
    # Equal variance
    e_dist_type, p_value = equal_variance_assumption(model, x_with_const, y, p_value_thresh, plt)
    if(plt):
        print('Equal variance:')
        print(e_dist_type, p_value)
    # Perfect collinearity
    has_perfect_collinearity = perfect_collinearity_assumption(x, plt)
    if(plt):
        print('Perfect collinearity:')
        print(has_perfect_collinearity)

    if is_linearity_found and autocorrelation is None and n_dist_type == 'normal' and e_dist_type == 'equal' and not has_perfect_collinearity:
        return True
    else:
        return False

def fit_and_get_rsquared_adj_test(x_train, x_test, y_train, y_test):
    '''pomoćna funkcija koja vraca fitovan model i prilagodjeni r^2 nad test skupom'''
    model = get_fitted_model(x_train, y_train)
    adj_r2 = get_rsquared_adj(model, x_test, y_test)
    return model, adj_r2

def fit_and_get_rsquared_adj_test_tree(x_train, x_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(x_train, y_train)
    n = len(y_test)
    p = x_test.shape[1]
    r_squared = model.score(x_test, y_test)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return model, adjusted_r_squared

if __name__ == '__main__':
    df = pd.read_csv('winequality-red.csv', sep=',')
    
    x = df.drop(columns=['kvalitet'])
    y = df['kvalitet']
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const).fit()
    print(model.summary())

    if are_assumptions_satisfied(model, x_train, y_train, p_value_thresh=0.05, plt=True):
        print('sve pretpostavke su zadovoljene')
    else:
        print('bar jedna pretpostavka nije zadovoljena')

    #Izračunavanje koeficijenata modela kroz DataFrame objekat koji sadrži informacije o koeficijentima linearne regresije
    coefficients = pd.DataFrame({
        'Feature': x_with_const.columns,    #nazivi svih kolona
        'Coefficient': model.params         #parametri modela koji su naučeni tokom fitovanja
    })

    coefficients = coefficients.iloc[1:]

    # Vizualizacija koeficijenata
    plt.figure(figsize=(10, 6))
    plt.bar(coefficients['Feature'], coefficients['Coefficient'])
    plt.xticks(rotation=35, ha='right')
    plt.xlabel('Hemijska karakteristika')
    plt.ylabel('Koeficijent')
    plt.title('Uticaj hemijskih karakteristika na ocenu kvaliteta vina')
    plt.show()

    cols = ['fiksnaKiselost', 'citricnaKiselost', 'ostatakSecera', 'gustina', 'pH', 'slobodniSumporDioksid']
    x_train_new, x_val_new = x_train.drop(columns=cols), x_val.drop(columns=cols)
    model, adj_r2 = fit_and_get_rsquared_adj_test(x_train_new, x_val_new, y_train, y_val)
    print(model.summary(), "\n", "prilegodjeni R^2 mera na test skupu - regresija: ", adj_r2)
    if not are_assumptions_satisfied(model, x_train_new, y_train, plt=True): print('pretpostavke nisu zadovoljene')

    print('ali posto imamo mnogo podataka mozemo smatrati da su sve pretpostavke zadovoljene')

    print(check_for_missing_values(df))
    #nema nedostajucih vrednosti

    df = pd.read_csv('winequality-red.csv', sep=',')
    x = df.drop(columns=['kvalitet'])
    y = df['kvalitet']
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

    #Stablo odlucivanja
    model, r_squared = fit_and_get_rsquared_adj_test_tree(x_train, x_val, y_train, y_val)
    print(f'prilagodjeni R^2 mera na test skupu - Stablo odlucivanja: {r_squared}')

    print('Veca mera prilagodjenog R^2 govori koja je opcija bolja')
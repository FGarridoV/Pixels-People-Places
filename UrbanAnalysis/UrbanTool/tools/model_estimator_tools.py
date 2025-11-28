import statsmodels.api as sm  
from scipy import stats

#def estimate_ols_model(df, y_col, X_cols):
#    y = df.loc[:,[y_col]]
#    X = df.loc[:,X_cols]
#    ols_mod = sm.OLS(y, X).fit() 
#    return ols_mod  

def estimate_ols_model(df, y_col, X_cols, add_constant = False, standarized = False):

    if standarized:
        add_constant = False
        df_z = df.loc[:,[y_col] + X_cols].apply(stats.zscore)
        y = df_z.loc[:,[y_col]]
        X = df_z.loc[:,X_cols].copy()
    else:
        y = df.loc[:,[y_col]]
        X = df.loc[:,X_cols].copy()
    
    X = X.fillna(0) # Checker where this nana comes from

    if add_constant:
        X = sm.add_constant(X)
    ols_mod = sm.OLS(y, X).fit()        #predictions = ols_mod.predict(X) 
    return ols_mod  
    

   
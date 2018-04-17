def window_dfdata(datatype, chosen_data, window_size):
    """
    Testing is going to be executed on real-time data.
    datatype: String(Pick,Drop, Noise).
    chosen_data: dataframe to be splitted into windows.
    """
    ind = 0
    data = []
    data = pd.concat(chosen_data)
    final_windows_df = pd.DataFrame()
    
    df_dict = {"pick" : pd.DataFrame(), "drop": pd.DataFrame(), "noise": pd.DataFrame()}

    index = 0
    while ((index + window_size) < len(data)):
        rows = data[index:(index + window_size)]
        statistics = rows.astype(int).describe()[0]
        picked = 0
        if "pick" in datatype.lower():
            picked = 1
        elif "noise" in datatype.lower():
            picked = 2
        
    
        df_dict[datatype] = df_dict[datatype].append({'mean': statistics['mean'],
                    'std': statistics['std'], 
                    'rms': rms,
                    'minimum':  statistics['min'],
                    'percentile_25' : statistics['25%'], 
                    'percentile_50' : statistics['50%'],
                    'percentile_75' : statistics['75%'],
                    'max' : statistics['max'],
                    'cov': 1.0 * np.mean(each.astype(int)) / statistics['std'],              
                    'kurtosis' :  each.astype(int).kurt().values[0],  
                    'skewness' :  each.astype(int).skew().values[0],
                    'variance' :  each.astype(int).var().values[0],             
                    'picked': picked
                    }, ignore_index=True)
        
        
        index += window_size
        
    labeled_df = df_dict['pick'].append(df_dict['drop']).append(df_dict['noise'])
    labeled_df = labeled_df.sample(frac=1).reset_index(drop=True)
    
    return labeled_df
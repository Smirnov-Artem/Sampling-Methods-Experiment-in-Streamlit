import pandas as pd
import numpy as np
import random
import networkx as nx
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from scipy.stats import jarque_bera, shapiro
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#from reliability.Reliability_testing import sequential_sampling_chart

# some methods are based on the https://medium.com/@ayanchowdhury00/sampling-techniques-and-its-implementation-in-python-487995b9399c

def generate_data(sorting, n, constant_value, betas, degrees, beta_dummy_checks, beta_dummy_vars, random_seed=100):
    np.random.seed(random_seed)  # Setting seed for reproducibility
    X = np.random.randn(n, len(betas))
    errors = np.random.randn(n)
    y = constant_value + np.dot(X, betas) + errors
    Y = pd.DataFrame({'Y': y})
    simulated_data = pd.concat([Y, pd.DataFrame(X, columns=['X'+str(i+1) for i in range(X.shape[1])])], axis=1)
    prefix = "X"
    for i in range(0,len(beta_dummy_checks)):
        if beta_dummy_checks[i]>=0:
            percentile = np.percentile(simulated_data[beta_dummy_vars[i]], beta_dummy_checks[i]*100)
            simulated_data[f"{prefix}{i+1}"] = np.where(simulated_data[beta_dummy_vars[i]] < percentile, 0.0,  np.where(simulated_data[beta_dummy_vars[i]] >= percentile, 1.0, 0))
    for i in range(0,len(degrees)):
        if degrees[i]!=1:
            simulated_data[f"{prefix}{i+1}"] = (simulated_data[f"{prefix}{i+1}"])**degrees[i]
    if sorting == True:
        simulated_data = simulated_data.sort_values(by='Y', ascending=False)
        simulated_data = simulated_data.reset_index().drop('index', axis=1)
        Y = Y.sort_values(by='Y', ascending=False)
        Y = Y.reset_index().drop('index', axis=1)
    return simulated_data, Y, X

def homoscedasticity_normality(model, simulated_data):
    my_dict = {}
    # Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, sm.add_constant(simulated_data.iloc[:, 1:]))
    my_dict["Breusch-Pagan test p-value:"] = bp_test[1]
    # White test
    white_test = het_white(model.resid, sm.add_constant(simulated_data.iloc[:, 1:]))
    my_dict["White test p-value:"] = white_test[1]
    # Jarque-Bera test
    jb_test = jarque_bera(model.resid)
    my_dict["Jarque-Bera test p-value:"] = jb_test[1]
    # Shapiro-Wilk test
    sw_test = shapiro(model.resid)
    my_dict["Shapiro-Wilk test p-value:"] = sw_test[1]

    return my_dict

def network_sampling(simulated_data_NETWORK, len_for_a_sample):
    G = nx.Graph()
    for _, row in simulated_data_NETWORK.iterrows():
        G.add_node(row['family_ID'])
        columns_starting_with_X = [col for col in simulated_data_NETWORK.columns if col.startswith('X')]
        for X_ in columns_starting_with_X:
            G.nodes[row['family_ID']][X_] = row[X_]

    # Sample families from the network graph
    sampled_nodes = random.sample(list(G.nodes()), len_for_a_sample)  # random N families

    sampled_data = simulated_data_NETWORK[simulated_data_NETWORK['family_ID'].isin(sampled_nodes)]

    return sampled_data


def sampling(sorting, model, simulated_data, sampling_technique, X, Y, std_err, sample_size, n_percent):
    if sampling_technique == "SRS":
        #defining the population from where sample will be created
        population = list(range(0, len(X))) 
        #defining the size of sample
        #perform simple random sampling by using the random.sample() function
        sample = random.sample(population, sample_size) 
        X_SAMPLING = sm.add_constant(simulated_data.iloc[:, 1:]).iloc[sample]
        model_SAMPLING = sm.OLS(Y.iloc[sample], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y.iloc[sample])

    elif sampling_technique == "BERN":
        theta = sample_size/len(X)
        # Generate Bernoulli samples
        num_samples = len(X)
        bernoulli_samples = [random.choices([0, 1], weights=[1-theta, theta])[0] for _ in range(num_samples)]
        sample = (np.argwhere(bernoulli_samples)).ravel()
        X_SAMPLING = sm.add_constant(simulated_data.iloc[:, 1:]).iloc[sample]
        model_SAMPLING = sm.OLS(Y.iloc[sample], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y.iloc[sample])
        
    elif sampling_technique == "STR":
        # Specify the stratification variable
        stratify_by = 'X2'
        # Split the data into training and testing sets, with stratification
        train_SAMPLING, test_SAMPLING = train_test_split(simulated_data, test_size=1-n_percent, stratify=simulated_data[stratify_by])
        sample = train_SAMPLING
        X_SAMPLING = sm.add_constant(sample.iloc[:, 1:])
        model_SAMPLING = sm.OLS(Y.loc[sample.index], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y.loc[sample.index])
        
    elif sampling_technique == "SIC":
        # create a list of population data
        population_data = list(np.asarray(simulated_data.index))
        # set the desired cluster size
        cluster_size = 10
        # randomly select a starting point for the first cluster
        starting_point = 0#random.randint(0, cluster_size + 1)
        # create a list to store the sampled data
        sampled_data = []
        # loop through the population data by clusters of size cluster_size
        for i in range(starting_point, len(population_data), cluster_size):
            # append the data from the current cluster to the sampled data list
            sampled_data.append(population_data[i:i+cluster_size])
        SIC_k_size = int(round(len(sampled_data)*n_percent, 0))
        random_elements = random.choices(sampled_data, k=SIC_k_size)
        random_elements = np.ravel(np.unique(random_elements))
        X_SAMPLING = sm.add_constant(simulated_data.iloc[:, 1:]).iloc[random_elements]
        model_SAMPLING = sm.OLS(Y.iloc[random_elements], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y.iloc[random_elements])

    elif sampling_technique == "TSC":
        # create a list of population data
        population_data = list(np.asarray(simulated_data.index))
        # set the desired cluster size
        cluster_size = 10
        # randomly select a starting point for the first cluster
        starting_point = 0#random.randint(0, cluster_size + 1)
        # create a list to store the sampled data
        sampled_data = []
        # loop through the population data by clusters of size cluster_size
        for i in range(starting_point, len(population_data), cluster_size):
            # append the data from the current cluster to the sampled data list
            sampled_data.append(population_data[i:i+cluster_size])
        TSC_k_size = int(round(len(sampled_data)*n_percent*2, 0))
        random_elements = random.choices(sampled_data, k=TSC_k_size)
        random_elements = np.ravel(np.unique(random_elements))
        random_elements_TSC = random.sample(list(random_elements), int(round(len(random_elements)/2,0)))
        X_SAMPLING = sm.add_constant(simulated_data.iloc[:, 1:]).iloc[random_elements_TSC]
        model_SAMPLING = sm.OLS(Y.iloc[random_elements_TSC], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y.iloc[random_elements_TSC])

    elif sampling_technique == "SYS":
        # Define the population
        population = np.arange(0, len(simulated_data))
        # Define the sample size and sampling interval
        # We can provide sample size and sampling interval as per user
        sampling_interval = len(population) // sample_size
        # Define the starting point of the sample
        # refer to https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html for random.randint()
        start_point = np.random.randint(0, sampling_interval)
        # Perform systematic sampling
        sample = population[start_point::sampling_interval]
        X_SAMPLING = sm.add_constant(simulated_data.iloc[:, 1:]).iloc[sample]
        model_SAMPLING = sm.OLS(Y.iloc[sample], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y.iloc[sample])
    
    elif sampling_technique == "QUOTA":
        shuffled_indices = simulated_data.index.to_list()
        np.random.shuffle(shuffled_indices)
        indeces_0s = []
        indeces_1s = []
        for index in shuffled_indices:
            row = simulated_data.loc[index]
            if row['X2'] == 0.0:
                if len(indeces_0s) < int((simulated_data['X2'] == 0.0).sum() * n_percent):
                    indeces_0s.append(index)
            elif row['X2'] == 1.0:
                if len(indeces_1s) < int((simulated_data['X2'] == 1.0).sum() * n_percent):
                    indeces_1s.append(index)
            if len(indeces_0s) >= int((simulated_data['X2'] == 0.0).sum() * n_percent) and \
            len(indeces_1s) >= int((simulated_data['X2'] == 1.0).sum() * n_percent):
                break
        sample = (np.ravel([indeces_0s + indeces_1s]))
        X_SAMPLING = sm.add_constant(simulated_data.iloc[:, 1:]).iloc[sample]
        Y_SAMPLING = Y.loc[sample]
        model_SAMPLING = sm.OLS(Y_SAMPLING, X_SAMPLING).fit(cov_type='HC3')
        Y_hat = (Y_SAMPLING)

    elif sampling_technique == "NETWORK":
        simulated_data_NETWORK = simulated_data.copy(deep=False)
        simulated_data_NETWORK['family_ID'] = pd.Series((np.repeat(np.arange(int(len(X))), np.random.randint(1, 7, size=int(len(X)))))[0:len(X)])
        # Shuffling the dataframe to randomize the family members
        simulated_data_NETWORK = simulated_data_NETWORK.sample(frac=1)
        if sorting == True:
            simulated_data_NETWORK = simulated_data_NETWORK.sort_values(by='Y', ascending=False)
        simulated_data_NETWORK = simulated_data_NETWORK.reset_index().drop('index', axis=1)
         #Let's build a network graph based on family relationships
        sampled_data = []
        sampled_data = network_sampling(simulated_data_NETWORK, int(round(len(simulated_data)*n_percent/3.5, 0)))
        X_SAMPLING = sm.add_constant(sampled_data.iloc[:, 1:len(sampled_data.columns)-1])
        model_SAMPLING = sm.OLS(sampled_data['Y'], X_SAMPLING).fit(cov_type='HC3')
        Y_hat = sampled_data['Y']
    
    mu_hat = np.mean(Y_hat)
    mu = np.mean(Y)
    bias = (mu_hat-mu)/mu * 100

    squared_diff = [(x - mu) ** 2 for x in np.array(Y_hat)]
    if type((sum(squared_diff) / len(Y_hat))) != np.float64:
        variance = (sum(squared_diff) / (len(Y_hat)-1))[0]
    else:
        variance = (sum(squared_diff) / (len(Y_hat)-1))

    summary_SAMPLING = model_SAMPLING.summary()
    rmse_SAMPLING = np.sqrt(mean_squared_error(Y, model_SAMPLING.predict(sm.add_constant(simulated_data.iloc[:, 1:]))))
    std_err_SAMPLING = model_SAMPLING.bse

    #homoscedasticity_normality(model_SAMPLING, X_SAMPLING)
    Coefficient_Difference = []
    Standard_Error_of_Difference = []
    t_statistic_column = []
    p_value_column = []
    # Degrees of freedom
    df = min(len(X), len(X_SAMPLING)) - 1

    for i in range(0,len(model.params)):
        coef_model1 = model.params[i]
        se_model1 = std_err[i]
        coef_model2 = model_SAMPLING.params[i]
        se_model2 = std_err_SAMPLING[i]
        # The difference in coefficients
        coef_diff = coef_model1 - coef_model2
        # The standard error of the difference
        se_diff = np.sqrt(se_model1**2 + se_model2**2)
        # t-statistic
        t_statistic = coef_diff / se_diff
        # p-value
        p_value = 2 * (1 - t.cdf(np.abs(t_statistic), df))
        Coefficient_Difference.append(coef_diff)
        Standard_Error_of_Difference.append(se_diff)
        t_statistic_column.append(t_statistic)
        p_value_column.append(p_value)
    df1 = pd.DataFrame({'coef_initial': model.params, 'coef_SAMPLING': model_SAMPLING.params, 'Coefficient_Difference': Coefficient_Difference, 'Standard_Error_of_Difference': Standard_Error_of_Difference, 't_statistic_column': t_statistic_column, 'p_value_column': p_value_column})
    return df1, rmse_SAMPLING, summary_SAMPLING, len(X_SAMPLING), sampling_technique, bias, variance
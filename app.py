import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import sampling_methods as methods
import plotly.graph_objs as go
import time
import os

def random_integer(minimum, maximum):
    # Generate a random integer in the range [minimum, maximum]
    difference = maximum - minimum
    random_bytes = int.from_bytes(os.urandom(4), byteorder='big')
    return minimum + random_bytes % (difference + 1)

def random_float():
    # Generate a random float in the range [0.0, 1.0)
    random_bytes = int.from_bytes(os.urandom(4), byteorder='big')
    return random_bytes / (1 << 32)

def random_choice(choices):
    # Choose a random element from the given list of choices
    index = random_integer(0, len(choices) - 1)
    return choices[index]

def main():
    # Page title
    st.title("Experiment for sampling methods")
    # Sidebar
    st.sidebar.header("Experiment type")
    genre = st.sidebar.radio(
    "Experiment type",
    ["Automatically generated N-experiments", "Individually adjusted single experiment"], label_visibility="collapsed")
    if genre == "Individually adjusted single experiment":
        exp_status = True
        exp_value = 1
    else:
        exp_status = False
        exp_value = 10
    n_exp = st.sidebar.number_input("N of experiments:", value=exp_value, min_value=1, max_value=100000, format='%u', key=1, disabled=exp_status)
    st.sidebar.header("Data generation")
    if genre == "Individually adjusted single experiment":
        n = st.sidebar.number_input("N:", value=1000, min_value=100, max_value=100000, format='%u', key="_N_")
        n_s = st.sidebar.number_input("n:", value=100, min_value=40, max_value=1000, format='%u', key="_n_")
        n_percent = n/n_s/100
    else:
        N_min = st.sidebar.number_input("N min:", value=500, min_value=100, max_value=100000, format='%u', key="_N_min")
        N_max = st.sidebar.number_input("N max:", value=10000, min_value=100, max_value=100000, format='%u', key="_N_max")
        n_percent = st.sidebar.slider("n/N:", min_value=0.01, max_value=0.25, step=0.01, key="n_percent")
    sorting = st.sidebar.checkbox("Sort the Y, descending order (affects different methods, including those where clusters are used)", key='sorting_key', value=False)
    st.write("Selected sampling methods:")
    method_names = st.multiselect("Selected sampling methods:", ["SRS", "BERN", "SIC", "TSC", "SYS", "NETWORK", "SEQUENTIAL",  "STR", "QUOTA"], key="methods", default=["SRS", "BERN", "SIC", "TSC", "SYS", "NETWORK", "SEQUENTIAL",  "STR", "QUOTA"], label_visibility="collapsed")

    final_dict_list = []
    places = [i for i in range(1, len(method_names)+1)]
    for i in range(1,len(method_names)+1):
        new_dict = {method_names[i]: {"count": 0} for i in range(len(places))}
        final_dict_list.append(new_dict)
    
    final_biases = {method_name: [] for method_name in method_names}
    final_vars = {method_name: [] for method_name in method_names}

    t = 0
    if genre == "Individually adjusted single experiment":
        n_exp = 1
    progress_bar = st.progress(0)
    for n_ in range(0,n_exp):
        while t<n_exp:
            progress_bar.progress((t + 1) / n_exp)
            try:
                if genre == "Individually adjusted single experiment":
                    pass
                else:
                    n = random_integer(N_min, N_max)
                    n_s = int(n_percent*n)
                if genre == "Individually adjusted single experiment":
                    n_variables = st.sidebar.number_input("Number of variables:", value=9, min_value=1, max_value=9, format='%u', key=str(n_)+"n_variables")
                else:
                    minimum = 4
                    maximum = 9
                    n_variables = random_integer(minimum, maximum)
                if genre == "Individually adjusted single experiment":
                    const = st.sidebar.number_input("Constant:", value=100.00, key="const")
                else:
                    const = random_integer(-100,100) * random_float()
                prefix = "X_"
                variables = {}
                for i in range(n_variables):
                    var_name = f"{prefix}{i+1}"
                    var_value = "variable_key_"+str(i)
                    variables[var_name] = var_value
                betas = []
                beta_degrees = []
                beta_dummy_checks = []
                beta_dummy_vars = []
                st.sidebar.markdown(
                    """
                    <style>
                        .sidebar .sidebar-content {
                            padding-top: 10px; /* Adjust padding to position the line vertically */
                        }
                        .sidebar-separator {
                            border-top: 2px solid #536474;
                            margin: 10px 0;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                predefined_betas = [15.0, 5.0, 5.0, 0.1, 0.0, -0.1, -5.0, -10.0, -15.0, -20.0]
                predefined_betas_dict = {}
                conti_dummys = []
                for i in range(0,10):
                    var_name = f"{prefix}{i+1}"
                    var_value = predefined_betas[i]
                    predefined_betas_dict[var_name] = var_value
                for variable in variables:
                    if genre == "Individually adjusted single experiment":
                        st.sidebar.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True) # horizontal grey line
                    title = "**Beta for X" + str(variable)[-1:] + "**"
                    if genre == "Individually adjusted single experiment":
                        if str(variable)[-1:]!=2:
                            beta = st.sidebar.number_input(title, key=variable, value=predefined_betas_dict[variable])
                        else:
                            beta = 2
                    else:
                        beta = random_integer(1,20) * random_float() - random_integer(1,10)
                    if genre == "Individually adjusted single experiment":
                        if variable!='X_2':
                            conti_dummy = st.sidebar.radio('Type: ', ["Continious", "Dummy"],key=variable+"dummy")
                        else:
                            conti_dummy = "Dummy"
                            conti_dummys.append(conti_dummy)
                    else:
                        if n_variables>6 and variable!='X_1' and variable!='X_2':
                            conti_dummy = random_choice(["Continious", "Dummy"])
                            conti_dummys.append(conti_dummy)
                        if variable=='X_2':
                            conti_dummy = "Dummy"
                            conti_dummys.append(conti_dummy)
                        else:
                            conti_dummy = "Continious"
                    if conti_dummy == "Continious":
                        if genre == "Individually adjusted single experiment":
                            beta_degree_check = st.sidebar.checkbox("Include degree", key=str(variable+"checkbox1"))
                        else:
                            beta_degree_check = True * np.random.randint(2)
                        if beta_degree_check == True:
                            title_degree = "Degree for X" + str(variable)[-1:]
                            if genre == "Individually adjusted single experiment":
                                beta_degree = st.sidebar.number_input(title_degree, value=2, min_value=1, max_value=5, key=str(variable+"degree"), format='%u')
                            else:
                                beta_degree = np.random.randint(1,5)
                            beta_degrees.append(beta_degree)
                        else:
                            beta_degrees.append(1)
                        beta_dummy_checks.append(-1)
                        beta_dummy_vars.append(-1)
                    else:
                        myList = list(["Y"])
                        myList.extend(variables.keys())
                        beta_dummy_var = "Y"
                        if genre == "Individually adjusted single experiment":
                            beta_dummy_check = st.sidebar.slider("Rule for dummy:", value=0.5, min_value=0.0, max_value=1.0, step=0.1, key=str(variable+"checkbox2"))
                        else:
                            beta_dummy_check = np.random.random_sample()
                        beta_dummy_checks.append(beta_dummy_check)
                        beta_dummy_vars.append(beta_dummy_var)
                        beta_degrees.append(1)
                    betas.append(beta)

                if genre == "Individually adjusted single experiment":
                    st.write("Underlying formula for the data generation:")
                formula = r"Y = "
                if const!=0:
                    formula += str(round(const,2))
                for i, (variable, beta) in enumerate(zip(variables, betas)):
                    if (round(beta,2))!=0.0:
                        if beta_dummy_checks[i] != -1:
                            variable = "D_"+variable[2]
                        if betas[i] >= 0.0 and const!=0.0:
                            formula += r" + "
                        if betas[i] >= 0.0 and const==0.0 and formula!=r"Y = ":
                            formula += r" + "
                        if beta_degrees[i] == 1:
                            formula += f"{str(round(beta,2))} \\cdot {variable}"
                        else:
                            power = {beta_degrees[i]}
                            formula += f"{str(round(beta,2))} \\cdot {variable}^{power}"
                simulated_data, Y, X = methods.generate_data(sorting, n, constant_value=const, betas=betas, degrees=beta_degrees, beta_dummy_checks=beta_dummy_checks, beta_dummy_vars=beta_dummy_vars)
                if genre == "Individually adjusted single experiment":
                    st.latex(formula + "+" + r"\epsilon")
                    with st.expander("See generated data", expanded=False):
                        st.dataframe(simulated_data, width=1000)
                model = sm.OLS(Y, sm.add_constant(simulated_data.iloc[:, 1:])).fit()
                if genre == "Individually adjusted single experiment":
                    with st.expander("See listings for initial OLS regression", expanded=False):
                            summary = model.summary()
                            st.write(summary)
                            # Root Mean Squared Error (RMSE)
                            rmse = np.sqrt(mean_squared_error(Y, model.predict(sm.add_constant(simulated_data.iloc[:, 1:]))))
                            st.write("RMSE:", rmse)
                            st.write(methods.homoscedasticity_normality(model, simulated_data))
                std_err = model.bse
                rmse_samplings = []
                N_samplings = []
                names = []
                biases = []
                std_var_Ys = []
                for method_name in method_names:
                    df, rmse_sampling, summary_sampling, N_sampling, name, bias, std_var_Y  = (methods.sampling(sorting, model, simulated_data, method_name, X, Y, std_err, n_s, n_percent))
                    (final_biases[str([item for item in final_biases if method_name in item][0])]).append(bias)
                    (final_vars[str([item for item in final_vars if method_name in item][0])]).append(std_var_Y)
                    biases.append(bias)
                    rmse_samplings.append(rmse_sampling)
                    N_samplings.append(N_sampling)
                    names.append(name)
                    std_var_Ys.append(std_var_Y)
                sorted_indices = np.argsort(rmse_samplings)
                rating = np.empty_like(rmse_samplings)
                rating[sorted_indices] = np.arange(1, len(rmse_samplings) + 1)
                df2 = pd.DataFrame({'names': names, 'rmse_samplings': rmse_samplings, 'N_samplings': N_samplings, 'Rating': rating, 'Mu_bias%': biases, "Var": std_var_Ys})
                if genre == "Individually adjusted single experiment":
                    st.write(df2)
                for k in range(0,len(final_dict_list)):
                    for method in method_names:
                        if int(df2[df2['names'] == method]['Rating']) == k+1:
                            final_dict_list[k][method]["count"] = final_dict_list[k][method]["count"]+1
                t += 1
            except:
                pass
            time.sleep(0.1)
    progress_bar.empty()
    # Aggregate counts for all dictionaries
    all_categories = set()
    category_counts = {}
    for first_dict in final_dict_list:
        for category, value in first_dict.items():
            all_categories.add(category)
            if category not in category_counts:
                category_counts[category] = []
            category_counts[category].append(value["count"])

    if genre == "Automatically generated N-experiments":
        # Create DataFrame for Plotly
        df = pd.DataFrame(category_counts, index=range(len(final_dict_list)))
        # Plotting the data
        fig1 = go.Figure()
        all_categories = sorted(all_categories, reverse=True)
        for category in all_categories:
            fig1.add_trace(go.Bar(x=(df.index)+1, y=df[category], name=category))

        # Add titles and labels
        fig1.update_layout(barmode='stack', xaxis_title='Ranking (1 - minimal RMSE, ' + str(len(method_names)) + ' - maximal RMSE)', yaxis_title='Absolute frequency', title='Comparative ranking of methods by RMSE levels for regressions on randomly generated data',
                                margin=dict(
                                    l=40,
                                    r=30,
                                    b=80,
                                    t=100,
                                ),
                                paper_bgcolor='rgb(240, 242, 246)',
                                plot_bgcolor='rgb(240, 242, 246)',
                                showlegend=True)

        # Display plot in Streamlit app
        st.plotly_chart(fig1)

        x_data = sorted(list(final_biases.keys()))
        y_data = []
        for method in x_data:
            y_ = final_biases[method]
            y_data.append(y_)
        fig2 = go.Figure()
        for xd, yd in zip(x_data, y_data):
                fig2.add_trace(go.Box(
                    y=yd,
                    name=xd,
                    boxpoints='suspectedoutliers',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker_size=2,
                    line_width=1)
                )
        fig2.update_layout(
            title='%Bias of the mean Y in generated datasets for each sample method',
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                dtick=5,
                gridcolor='rgb(255, 255, 255)',
                gridwidth=1,
                zerolinecolor='rgb(255, 255, 255)',
                zerolinewidth=2,
            ),
            margin=dict(
                l=40,
                r=30,
                b=80,
                t=100,
            ),
            paper_bgcolor='rgb(240, 242, 246)',
            plot_bgcolor='rgb(240, 242, 246)',
            showlegend=False
        )
        st.plotly_chart(fig2)


        x_data = sorted(list(final_vars.keys()))
        y_data = []
        for method in x_data:
            y_ = final_vars[method]
            y_data.append(y_)

        merged_list = sum(y_data, [])
        fig3 = go.Figure()
        for xd, yd in zip(x_data, y_data):
                fig3.add_trace(go.Box(
                    y=yd,
                    name=xd,
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker_size=2,
                    line_width=1)
                )
        result_list = list(range(0, int(max(merged_list)) + 1, 50))
        fig3.update_layout(
            title='Variance of Y in generated datasets for each sample method',
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                tickvals=result_list,  # Adjust these values as needed
                dtick=5,
                gridcolor='rgb(255, 255, 255)',
                gridwidth=1,
                zerolinecolor='rgb(255, 255, 255)',
                zerolinewidth=2,
            ),
            margin=dict(
                l=40,
                r=30,
                b=80,
                t=100,
            ),
            paper_bgcolor='rgb(240, 242, 246)',
            plot_bgcolor='rgb(240, 242, 246)',
            showlegend=False
        )
        st.plotly_chart(fig3)

if __name__ == "__main__":
    main()
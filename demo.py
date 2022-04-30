import pandas as pd
import streamlit as st
from fbprophet import Prophet, plot
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import logging
import sys
from sklearn.model_selection import train_test_split
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

DATA_PATH = 'data/taxi.csv'


@st.cache
def load_data():
    data = pd.read_csv(DATA_PATH)
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')

    return data


@st.experimental_memo
def prophet_data_clean(df, train):
    df_prophet = df.reset_index().rename(columns={"datetime": "ds", "num_orders": "y"})
    df_prophet_train = train.reset_index().rename(columns={"datetime": "ds", "num_orders": "y"})

    return df_prophet, df_prophet_train


def prophet_forecast(df, train, test):
    df, df_prophet_train = prophet_data_clean(df, train)

    prophetModel = Prophet()
    prophetModel.fit(df_prophet_train)

    future = prophetModel.make_future_dataframe(periods=test.shape[0], freq='10T')
    forecast = prophetModel.predict(future)

    if st.checkbox('Show raw FORECAST data'):
        st.subheader('Raw FORECAST data')
        st.write(forecast.iloc[-len(test):])

    ## ------ full plot
    st.write('Plot with forecasting')
    fig = prophetModel.plot(forecast)
    plot.add_changepoints_to_plot(fig.gca(), prophetModel, forecast)
    st.pyplot(fig)

    ## ------ only forecast plot

    fig, ax = plt.subplots()
    ax.plot(forecast.ds, forecast['yhat'])
    ax.set_xlim(left=train.datetime.iloc[-1000], right=test.datetime.max())
    ax.plot(test.datetime, test.num_orders, linewidth=0.1, color='green')
    fig.tight_layout()
    st.pyplot(fig)

    # ------ components plot
    st.write("Component wise forecast")
    components_plot = prophetModel.plot_components(forecast)
    st.write(components_plot)


def prophet_cross_validation(df, train, initial, horizon, period):
    df, df_prophet_train = prophet_data_clean(df, train)

    # initial = str(max(df_prophet_train.ds) - min(df_prophet_train.ds))
    # horizon = '1 hours'
    # period = '1 hours'

    initial = f'{initial} days'
    horizon = f'{horizon} minutes'
    period = f'{period} hours'

    m = Prophet().fit(df)

    df_cv = cross_validation(m,
                             initial=initial,
                             period=period,
                             horizon=horizon,
                             parallel="processes")

    df_p = performance_metrics(df_cv)
    # st.dataframe(df_p)

    fig1 = plot_cross_validation_metric(df_cv, metric='rmse')
    fig2 = plot_cross_validation_metric(df_cv, metric='mape')
    # st.write(fig)

    return df_p, fig1, fig2


@st.experimental_memo
def split(df, split_ratio):
    return train_test_split(df, test_size=split_ratio, shuffle=False)


def forecast_exp_smoothing(train, test):
    train_smth = train.set_index(train['datetime']).drop(columns='datetime')
    test_smth = test.set_index(test['datetime']).drop(columns='datetime')

    fit3 = SimpleExpSmoothing(train_smth, initialization_method="estimated").fit()

    predicted = fit3.predict(start=test_smth.index.min(), end=test_smth.index.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_smth.index, y=train_smth.num_orders,
                             mode='lines', name='train'))

    fig.add_trace(go.Scatter(
        x=test_smth.index, y=test_smth.num_orders,
        mode='markers', name='real test',
        marker=dict(
            color='LightSkyBlue',
            size=2,
        ),
    ))

    fig.add_trace(go.Scatter(x=predicted.index, y=predicted.to_list(),
                             mode='lines', name='predicted', line=dict(color="red")))

    fig.update_layout(title='Exponential smoothin')
    st.plotly_chart(fig)


if __name__ == "__main__":
    st.title('Hourly Taxi pickup forecast')

    window_selection_c = st.sidebar.container()  # create an empty container in the sidebar
    window_selection_c.markdown("## Insights")

    df_taxi = load_data()

    split_ratio = st.select_slider('Train/test split ratio:', options=[0.1, 0.25, 0.33])

    train, test = split(df_taxi, split_ratio)

    if st.checkbox('Show raw data'):
        st.subheader('Input Raw data')
        st.write(df_taxi)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_taxi.datetime,
                             y=df_taxi.num_orders,
                             mode='lines',
                             ))

    fig.add_vline(x=train.datetime.max(), line_width=4, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Taxi pickup number",
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        xaxis_title='time',
        yaxis_title=f"number",
    )

    st.plotly_chart(fig, use_container_width=True)

    option = window_selection_c.selectbox('Select model for prediction', ['Prophet', 'Exponential smoothing'],
                                          index=0)

    if option == 'Prophet':
        with st.spinner('Wait for prophet forecast. It will continue less than 40 seconds ...'):
            prophet_forecast(df_taxi, train, test)

        st.title('Cross validation metrics with default parameters')

        window_selection_c.markdown("### Cross validation Prophet")

        train_test_forecast_c = st.sidebar.container()

        initial = train_test_forecast_c.number_input("Init values days", min_value=30, max_value=180,
                                                     value=int(
                                                         str(max(train.datetime) - min(train.datetime)).split()[0]),
                                                     step=1)
        horizon = train_test_forecast_c.number_input("Horizonte minutes (60 was asked in pdf)",
                                                     min_value=30, max_value=180, value=60, step=10)
        period = train_test_forecast_c.number_input("Period hours", min_value=1, max_value=100, value=25, step=1)

        train_test_forecast_c.button(
            label="Start cross validation",
            key='CROSS_VALIDATION'
        )

        if st.session_state.CROSS_VALIDATION:
            with st.spinner('Wait for cross validation... For 1 to 3 minutes'):
                metics_df, plot_metrics_rmse, plot_metrics_mape = prophet_cross_validation(df_taxi, train, initial,
                                                                                           horizon, period)

            st.pyplot(plot_metrics_rmse)
            st.pyplot(plot_metrics_mape)
            # st.write(metics_df)

    elif option == 'Exponential smoothing':
        forecast_exp_smoothing(train, test)

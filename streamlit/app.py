import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import os

# "../data/processed/data.csv"

def late_delivery_rate(y_true, y_pred):
    # This is the metric we want to optimize for
    late_deliveries = np.sum(y_pred < y_true)
    return late_deliveries / len(y_true)

def data_preprocess():
    #file_path = "causal-inference-marketplace/data/processed/data.csv"  #../data/processed/data.csv"
    #file_path = os.path.join(os.path.dirname(__file__),'..', 'data', 'dataset.csv')
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'data.csv')
    df = pd.read_csv(file_path, parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
    df['purchase_date_hour'] = df['order_purchase_timestamp'].dt.floor('H')
    df['gap_in_minutes_approved_and_ordered'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600
    df['order_purchase_date'] = pd.to_datetime(df['order_purchase_timestamp'].dt.date)
    df['dow'] = df['order_purchase_timestamp'].dt.day_of_week
    df['hour'] = df['order_purchase_timestamp'].dt.hour
    # df['time_of_day'] = df['order_purchase_timestamp'].dt.hour.map(assign_time_of_day)
    df['days_to_actual_delivery'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['days_to_actual_delivery_log'] = np.log(df['days_to_actual_delivery']+1)
    df['days_to_estimated_delivery'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
    df['is_december'] = 0
    df.loc[df['month'].isin([11,12]), 'is_december'] = 1
    df['is_summer'] = 0
    df.loc[df['month'].isin([5,6,7,8]), 'is_summer'] = 1
    df_clean = df.drop_duplicates(subset=['order_id'])

    df_clean['purchase_date_hour'] = df_clean['order_purchase_timestamp'].dt.floor('H')
    df_clean['week_start'] = (df['order_purchase_timestamp'] - pd.to_timedelta(df['order_purchase_timestamp'].dt.weekday + 1, unit='D')).dt.date
    df_clean['week_start'] = pd.to_datetime(df_clean['week_start'])

    weekly_data = df_clean.groupby(pd.Grouper(key='order_purchase_date', freq='W')).agg(
        count_orders=('order_purchase_date', 'count'),
        num_late_deliveries = ('is_delivery_late', 'sum')).reset_index()

    weekly_data['perc_late_deliveries'] = weekly_data['num_late_deliveries'] / weekly_data['count_orders']
    weekly_data['perc_late_deliveries'].fillna(0, inplace=True)

    df_clean = pd.merge(df_clean, weekly_data, how='left', left_on='week_start', right_on='order_purchase_date', suffixes=('', '_duplicate'))

    # Lagging the weekly percentage late deliveries variable to know the late delivery rate from the previous weeks
    for i in range(1,3):
        df_clean[f'perc_late_deliveries_lag_{i}'] = df_clean['perc_late_deliveries'].shift(i)
        df_clean.dropna(subset=[f'perc_late_deliveries_lag_{i}'], inplace=True)

    return df_clean



def predict(df_clean):

    file_path_model_lb = os.path.join(os.path.dirname(__file__), '..', 'results', 'models', 'gb_model_lb.joblib')
    file_path_model_ub = os.path.join(os.path.dirname(__file__), '..', 'results', 'models', 'gb_model_ub.joblib')

    gb_model_lb = joblib.load(file_path_model_lb)
    gb_model_ub = joblib.load(file_path_model_ub)

    features = [
                'is_summer',
                'freight_value',
                'distance_km',
                'month',
                'dow',
                'is_december',
                'is_delivery_late',
                'perc_late_deliveries_lag_1']
    
    target = 'days_to_actual_delivery_log'
    X, y = df_clean[features], df_clean[target]

    lower_bound = gb_model_lb.predict(X)
    upper_bound = gb_model_ub.predict(X)

    df_clean['days_to_estimated_delivery_model_lb'] = np.exp(lower_bound) - 1
    df_clean['days_to_estimated_delivery_model_ub'] = np.exp(upper_bound) - 1

    # Calculating late deliveries using our model
    df_clean['is_delivery_late_gb'] = np.where(df_clean['days_to_actual_delivery'] > df_clean['days_to_estimated_delivery_model_ub'], 1, 0)
    return df_clean


def plot_prediction():
    df_clean = data_preprocess()
    df_clean = predict(df_clean)


    weekly_data_model_current = df_clean.groupby(pd.Grouper(key='order_purchase_date', freq='W')).agg(
        count_orders=('order_purchase_date', 'count'),
        num_late_deliveries = ('is_delivery_late', 'sum'))

    weekly_data_model_gb= df_clean.groupby(pd.Grouper(key='order_purchase_date', freq='W')).agg(
        count_orders=('order_purchase_date', 'count'),
        num_late_deliveries = ('is_delivery_late_gb', 'sum'))

    weekly_data_model_current['perc_late_deliveries'] = weekly_data_model_current['num_late_deliveries'] / weekly_data_model_current['count_orders']
    weekly_data_model_gb['perc_late_deliveries'] = weekly_data_model_gb['num_late_deliveries'] / weekly_data_model_gb['count_orders']

    olist_ldr = late_delivery_rate(df_clean['days_to_actual_delivery'], df_clean['days_to_estimated_delivery']).round(3)
    new_model_ldr = late_delivery_rate(y_true=df_clean['days_to_actual_delivery'], y_pred=df_clean['days_to_estimated_delivery_model_ub']).round(3)

    # Create a new dataframe for Plotly visualization
    weekly_data_model_current_clean = weekly_data_model_current[(weekly_data_model_current['perc_late_deliveries'].notna()) & 
                                                            (weekly_data_model_current['perc_late_deliveries'] != 1)]
    weekly_data_model_gb_clean = weekly_data_model_gb[(weekly_data_model_gb['perc_late_deliveries'].notna()) & 
                                                    (weekly_data_model_gb['perc_late_deliveries'] != 1)]

    # Add a column for the model name to distinguish them
    weekly_data_model_current_clean['model'] = f'Olist Current Model: LDR {olist_ldr}'
    weekly_data_model_gb_clean['model'] = f'New Model (Gradient Boosting): LDR {new_model_ldr}'

    # Combine both datasets for plotting
    combined_data = pd.concat([weekly_data_model_current_clean[['perc_late_deliveries', 'model']],
                            weekly_data_model_gb_clean[['perc_late_deliveries', 'model']]])



    # Create the Plotly plot
    fig = px.line(combined_data, 
                x=combined_data.index, 
                y='perc_late_deliveries', 
                color='model', 
                title="Percentage of late deliveries by week",
                labels={'perc_late_deliveries': 'Percentage late deliveries', 'index': 'Week'})

    # Show the plot
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)


def main():


    # Set page config
    st.set_page_config(
        page_title="Project Findings",
        page_icon="📊",
        layout='wide'
    )


    st.title("🔍 Marketplaces: Delivery Delays & Ratings")

    col1, col2= st.columns([2,1])

    with col1:
        # Overview Section
        st.subheader("Late Deliveries: A Causal Impact on Ratings")
        st.write("""
                This project investigates the causal effect of late delivery on customer satisfaction ratings,
                using data provided by Olist, the largest department store in Brazilian marketplaces.
        """)


    num_orders, late_orders, rate_late_delivery, avg_rating = st.columns([3,3,3,3])

    with num_orders:
        st.metric(label="Number of Orders", value="114,841")

    with late_orders:
        st.metric(label="Number of Late Deliveries", value="7368")

    with avg_rating:
        st.metric(label="Average Rating", value="4.08 ⭐")

    with rate_late_delivery:
        st.metric(label="Late Delivery Rate", value="6.4%")

    #                     # URL of your GitHub repository
    # repo_url = "https://github.com/juanpi19/causal-inference-marketplace"

    #     # Display GitHub logo with a link
    # st.markdown(f"""
    #     [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30">]({repo_url})
    #     """, unsafe_allow_html=True)
   

    # Compact Expander Section
    col1, col2, col3= st.columns([1,2,1])

    # Divider
    st.divider()

    # First Two Findings Side by Side
    
    st.subheader("Uncovering the Causal Impact")
    col1, col3 = st.columns([3,2])
  
    with col1:
        st.write(
            """
            To investigate the causal impact of late deliveries on customer ratings, we employed the **Propensity Score Matching (PSM)** technique. 
            By matching late and on-time deliveries based on key factors like distance, season, product details, and value, 
            we isolated the effect of late deliveries.
            """
        )
        st.markdown("[Propensity Score Matching Notebook](https://github.com/juanpi19/causal-inference-marketplace/blob/main/notebooks/model-development.ipynb)")

    # with col2:
    #     st.graphviz_chart('''
    #                 digraph {
    #                     LateDelivery -> Rating
    #                 }
    #             ''')

    with col3:
        st.metric(label="**Impact of Late Delivery on Customer Ratings**", value="1.8")

        # st.graphviz_chart('''
        #                     digraph {
        #                         LateDelivery -> Rating
        #                     }
        #                 ''')

        # Optional: Add a callout for emphasis
        st.info(
            """
            **Insight:** Late deliveries significantly impact customer ratings, reducing them by an average of 1.8 stars.
            """
        )

    # Divider
    st.divider()

    # Third Finding / Recommendation
    st.subheader("🌟 Solution")
    col1, col2 = st.columns([1,1])

    #with col1:
    st.write("""
    **Problem:** We have quantified how much Late deliveries are impacting customer satisfaction.

    **Solution:**
    1. **Accurate Predictions:** Improve delivery time estimates using historical data and predictive modeling.
    2. **Confidence Intervals:** Use 95% confidence intervals to minimize underestimations.
    3. **Reduced Late Deliveries:** Aim to decrease the current 7.5% late delivery rate.

    By implementing these solutions, we can enhance customer experience and operational efficiency.
    """)

    # Divider
    st.divider()
    st.subheader("💵 Business Impact")

    # with col2:
    plot_prediction()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.express as px

# "../data/processed/data.csv"

def late_delivery_rate(y_true, y_pred):
    # This is the metric we want to optimize for
    late_deliveries = np.sum(y_pred < y_true)
    return late_deliveries / len(y_true)

def data_preprocess(file_path):
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
    file_path = "../data/processed/data.csv"
    df_clean = data_preprocess(file_path)
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


    st.title("🔍 Online Marketplace Analysis")

    col1, col2= st.columns([2,1])

    with col1:
        # Overview Section
        st.header("Impact of Late Deliveries on Customer Satisfaction")
        st.write("""
                This project investigates the causal effect of late delivery on customer satisfaction ratings,
                using data provided by Olist, the largest department store in Brazilian marketplaces.
        """)


        with st.expander("Learn more"):
            st.markdown("""
            **Detailed Project Background**

            Understanding the causal relationship between delivery delays and customer satisfaction is crucial for e-commerce platforms, 
            as it directly impacts customer retention, brand reputation, and long-term profitability. 
            While delivery performance metrics are routinely collected, establishing a true causal link between delays and customer satisfaction 
            presents unique challenges. 
                        
            Although randomized controlled trials (RCTs) represent the gold standard in causal inference, deliberately delaying 
            customer deliveries for experimental purposes would be both unethical and potentially damaging to business operations. 
            This necessitates the use of observational study methods to draw reliable causal conclusions from existing data.
            
                        
            This study leverages the rich dataset from Olist, Brazil's largest department store marketplace, 
            to estimate the causal effect of delivery delays on customer satisfaction ratings.
             We employ two complementary approaches: propensity score matching to simulate experimental conditions using observational data, 
            and graphical causal models to understand the underlying data-generating process. 
            This dual methodology allows us to not only quantify the impact of delays but also to understand the complex mechanisms through which delivery performance affects customer satisfaction.
            """)

                    # URL of your GitHub repository
        repo_url = "https://github.com/juanpi19/causal-inference-marketplace"

        # Display GitHub logo with a link
        st.markdown(f"""
        [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30">]({repo_url})
        """, unsafe_allow_html=True)

   
    # Compact Expander Section
    col1, col2, col3= st.columns([1,2,1])

    # Divider
    st.divider()

    # First Two Findings Side by Side
    
    st.subheader("Uncovering the Causal Impact")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write(
            """
            To investigate the causal effect of late deliveries on customer ratings, we employ a **Propensity Score Matching (PSM)** approach. 
            PSM allows us to control for confounding variables by matching late delivery cases with similar on-time delivery cases based on 
            key features, including delivery distance (in km), season, product category, freight value, and product size. By matching on 
            these attributes, we ensure a fair comparison, isolating the effect of late deliveries on customer ratings.

            Our analysis reveals a notable impact on customer ratings, shedding light on areas for improvement in the delivery process 
            to enhance customer satisfaction.
            """
        )
        st.markdown("[Propensity Score Matching Notebook](https://github.com/juanpi19/causal-inference-marketplace/blob/main/notebooks/model-development.ipynb)")
    with col2:
        st.metric(label="**Impact of Late Delivery on Customer Ratings**", value="1.8")

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

    with col1:
        st.write(
            """
            Late deliveries occur when the actual shipping date surpasses the estimated delivery date, creating a gap between customer 
            expectations and operational performance. To address this issue, we propose the following solution:

            **Develop Realistic Delivery Timeframes:** Use historical data to generate accurate delivery estimates that we can consistently 
            meet or exceed. 
            
            Currently, Olist's Expected Delivery model has a Late Delivery Rate of 7.5%, which serves as our benchmark. 
            Our goal is to improve this rate and reduce the frequency of late deliveries, ensuring better alignment with customer expectations.

            ### Predicting Estimated Delivery
            We predict the **days to actual delivery**, defined as the number of days from the order purchase date to the actual delivery date. 
            This prediction allows us to construct the "is_late_delivery" variable based on the logic:

            - **Late Delivery:** `Days to Actual Delivery > Days to Estimated Delivery`
            - **On-Time Delivery:** `Days to Actual Delivery ≤ Days to Estimated Delivery`

            ### Addressing Prediction Errors
            When predicting delivery times, the model may err in two ways:
            
            - **Overestimation:** The predicted delivery time is longer than the actual delivery time. This typically results in early deliveries, which customers often prefer.
            - **Underestimation:** The predicted delivery time is shorter than the actual delivery time, leading to late deliveries and unmet expectations.

            Since late deliveries are the primary concern, we propose estimating a **confidence interval for delivery timeframes**. 
            By using a 95% confidence interval, we aim to account for uncertainty in predictions, focusing on minimizing underestimations while maintaining operational feasibility.
            """
        )

    with col2:
        plot_prediction()

if __name__ == "__main__":
    gb_model_lb = joblib.load('../results/models/gb_model_lb.joblib')
    gb_model_ub = joblib.load('../results/models/gb_model_ub.joblib')
    main()
'''
This make_dataset.py file is intended to keep all the functions that we will use during development. A couple of rules to follow:

1. Let's write proper docstrings for each function.
2. Let's include the data type that the function returns. For example, if a function returns a dict, it should be written as: def func() -> dict
3. Let's use clear and concise function and parameter names
4. Let's break stuff as quickly as possible so we learn faster


Example of function:

    def add_numbers(a: int, b: int) -> int:
        """
        Add two numbers together.

        This function takes two integers and returns their sum.

        Parameters:
        a (int): The first integer to add.
        b (int): The second integer to add.

        Returns:
        int: The sum of the two input integers.
        """
    return a + b

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from make_dataset import load_data, merge_all_datasets
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

state_to_region = {
    'AC': 'North', 'AP': 'North', 'AM': 'North', 'PA': 'North', 'RO': 'North', 'RR': 'North', 'TO': 'North',
    'AL': 'Northeast', 'BA': 'Northeast', 'CE': 'Northeast', 'MA': 'Northeast', 'PB': 'Northeast', 'PE': 'Northeast', 'PI': 'Northeast', 'RN': 'Northeast', 'SE': 'Northeast',
    'GO': 'Central-West', 'MT': 'Central-West', 'MS': 'Central-West', 'DF': 'Central-West',
    'ES': 'Southeast', 'MG': 'Southeast', 'RJ': 'Southeast', 'SP': 'Southeast',
    'PR': 'South', 'RS': 'South', 'SC': 'South'
}

# ------------------------ Functions
# ------------------------ 1
'''
def handle_missing_values(df):
    """
    Handles missing values in the DataFrame as per specific instructions.
    """

    # Drop rows where 'seller_id' is missing
    df.dropna(subset=['seller_id'], inplace=True)

    #drop unnecesary columns
    columns_to_drop = [
    'geolocation_zip_code_prefix_x',
    'geolocation_zip_code_prefix_y',
    'geolocation_city_x',
    'geolocation_state_x',
    'geolocation_city_y',
    'geolocation_state_y']

    df.drop(columns=columns_to_drop, inplace=True)

    # Convert date columns to datetime format and calculate average time deltas
    date_columns = [
        'order_purchase_timestamp', 
        'order_approved_at', 
        'order_delivered_carrier_date', 
        'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    avg_purchase_to_approval = (df['order_approved_at'] - df['order_purchase_timestamp']).mean()
    avg_approval_to_carrier = (df['order_delivered_carrier_date'] - df['order_approved_at']).mean()
    avg_carrier_to_customer = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).mean()

    # Impute missing dates based on calculated averages
    df['order_approved_at'] = df.apply(
        lambda row: row['order_purchase_timestamp'] + avg_purchase_to_approval 
        if pd.isna(row['order_approved_at']) else row['order_approved_at'],
        axis=1
    )
    df['order_delivered_carrier_date'] = df.apply(
        lambda row: row['order_approved_at'] + avg_approval_to_carrier 
        if pd.isna(row['order_delivered_carrier_date']) else row['order_delivered_carrier_date'],
        axis=1
    )
    df['order_delivered_customer_date'] = df.apply(
        lambda row: row['order_delivered_carrier_date'] + avg_carrier_to_customer 
        if pd.isna(row['order_delivered_customer_date']) else row['order_delivered_customer_date'],
        axis=1
    )
    df['order_delivered_customer_date'] = df.apply(
        lambda row: min(row['order_delivered_customer_date'], row['order_estimated_delivery_date']) 
        if not pd.isna(row['order_delivered_customer_date']) else row['order_delivered_customer_date'],
        axis=1
    )

    # Replace review-related columns' NaNs with 'no review' or 1000000000 for review_score
    review_columns = [
        'review_score', 'review_id', 'review_comment_title', 
        'review_comment_message', 'review_creation_date', 'review_answer_timestamp'
    ]
    df[review_columns] = df[review_columns].fillna('no review')
    df['review_score'] = df['review_score'].replace('no review', 1000000000)

    # Impute geolocation columns using KNN Imputer
    geolocation_columns_x = ['geolocation_lat_x', 'geolocation_lng_x']
    geolocation_columns_y = ['geolocation_lat_y', 'geolocation_lng_y']
    knn_imputer = KNNImputer(n_neighbors=5)
    df[geolocation_columns_x] = knn_imputer.fit_transform(df[geolocation_columns_x])
    df[geolocation_columns_y] = knn_imputer.fit_transform(df[geolocation_columns_y])

    # Drop unnecessary columns
    #df.drop(columns=['review_score_numeric'], inplace=True)

    # Replace NaNs in 'product_photos_qty' with '0'
    df['product_photos_qty'].fillna('0', inplace=True)

    # Drop rows with missing product dimensions or weights
    df.dropna(subset=['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'], inplace=True)

    # Impute product details using a helper function to find matching values
    def find_matching_row_values(df, row, target_column):
        match = (df[match_columns] == row[match_columns].values).all(axis=1)
        matching_values = df.loc[match & df[target_column].notna(), target_column]
        return matching_values.mode().iloc[0] if not matching_values.empty else None

    match_columns = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    for target_column in ['product_category_name', 'product_name_lenght', 'product_description_lenght']:
        df[target_column] = df.apply(
            lambda row: find_matching_row_values(df, row, target_column) 
            if pd.isna(row[target_column]) else row[target_column],
            axis=1
        )

    # Fill remaining NaNs in product fields with high values or 'unknown'
    df['product_name_lenght'] = pd.to_numeric(df['product_name_lenght'], errors='coerce')
    df['product_description_lenght'] = pd.to_numeric(df['product_description_lenght'], errors='coerce')
    df['product_name_lenght'].fillna(100000, inplace=True)
    df['product_description_lenght'].fillna(100000, inplace=True)
    df['product_category_name'].fillna('unknown', inplace=True)

    # Calculate and impute customer experience based on rolling averages
    df = df.sort_values(by=['customer_id', 'review_answer_timestamp'])
    df['customer_experience'] = df.groupby('customer_id')['review_score'].apply(rolling_mean_customer).reset_index(level=0, drop=True)
    df['customer_experience'].fillna(df['review_score'], inplace=True)

    # Calculate and impute seller experience
    df = df.sort_values(by=['seller_id', 'review_answer_timestamp'])
    df['seller_avg_rating'] = df.groupby('seller_id')['review_score'].apply(rolling_mean_seller).reset_index(level=0, drop=True)
    df['seller_avg_rating'].fillna(df['review_score'], inplace=True)
    return df'''

# ------------------------ 2
def rolling_mean_customer(group):
    filtered_reviews = group[group <= 5]
    expanding_mean = filtered_reviews.expanding().mean()
    return expanding_mean.reindex(group.index).ffill().bfill()

# ------------------------ 3
def rolling_mean_seller(group):
    filtered_reviews = group[group <= 5]
    expanding_mean = filtered_reviews.expanding().mean()
    return expanding_mean.reindex(group.index).ffill().bfill()
    
#----------------------------------- Handling Missing Values
#----- product_category
def get_features_and_target():
    '''Gets features and target for RF model'''
    features = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'seller_id_encoded', 'month']
    target = 'product_category_name_encoded'
    return features, target

def encode_column(df, column):
    '''Encode Column in dataframe'''
    df2 = df.copy()
    le = LabelEncoder()
    df2[f'{column}_encoded'] = le.fit_transform(df2[column])
    return df2, le

def train_random_forest_model(X, y):
    '''Fits Random Forest model'''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    rf1 = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rf1.predict(X_val)
    print("Random Forest accuracy on training set: ", accuracy_score(y_val, y_pred)) 
    rf2 = RandomForestClassifier().fit(X, y)
    return rf2

def impute_product_category(df):
    '''Imputes Product Category column'''
    features, target = get_features_and_target()
    df, le = encode_column(df, 'product_category_name')
    df, le2 = encode_column(df, 'seller_id')
    df.dropna(subset=features + [target], inplace=True)

    # Model
    missing_val_index_lst = df[df['product_category_name'].isna()].index
    impute_data = df.loc[missing_val_index_lst][features] 
    df_clean = df[~df.index.isin(missing_val_index_lst)]
    X, y = df_clean[features], df_clean[target]
    rf_model_full = train_random_forest_model(X, y)
    imputed_product_category = rf_model_full.predict(impute_data)

    df.loc[missing_val_index_lst, 'product_category_name_encoded'] = imputed_product_category
    df['product_category_name'] = le.inverse_transform(df['product_category_name_encoded'])
    print(f"{len(missing_val_index_lst)} have been imputed")
    return df

#----- freight_values
def drop_na(df, columns):
    """Drop rows with missing values in specified columns and return a new DataFrame"""
    df_copy = df.copy()
    df_copy.dropna(subset=columns, inplace=True)
    return df_copy

def handle_missing_values(df):
    '''this function handles final dataset missing values'''
    df['month'] = df['order_purchase_timestamp'].dt.month
    # Keeping Delivered and Shipped orders only
    df = df[df['order_status'].isin(['delivered', 'shipped'])]
    # Dropping null values
    df = drop_na(df, 'freight_value')
    # Impute product category
    df = impute_product_category(df)
    return df


# ------------------------------------------------------------- 4. harvesine
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude in decimal degrees.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371  # Radius of Earth in kilometers
    distance = r * c
    return distance

#------------------------------------------------------------------- 4. preprocessing

def preprocessing(df, state_to_region):
    '''
    takes raw dataset and preprocess it
    '''
    df['region'] = df['customer_state'].map(state_to_region)  
    df['Product_weight_kg'] = df['product_weight_g']/1000
    df['Product_category'] = df['product_category_name']

    df['Product_size'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df['No_photos'] = df['product_photos_qty']
    df['Product_price'] = df['price']

    df['late_delivery_in_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_delivery_late'] = np.where(df['late_delivery_in_days'] > 0, 1, 0)
    df = df[~df['order_delivered_customer_date'].isna()]
    df['Rating']= df['review_score']

    df['distance_km'] = df.apply(lambda row: haversine(row['geolocation_lat_x'], row['geolocation_lng_x'], row['geolocation_lat_y'], row['geolocation_lng_y']), axis=1)
    df.dropna(subset=['distance_km'], inplace=True)
    return df


# ------------------------------------------------ Main
def main():
    start = time.time()
    #----------------- Loading Data
    data_dict = load_data()
    olist_customers_df = data_dict['olist_customers_df']
    olist_geolocation_df = data_dict['olist_geolocation_df']
    olist_order_items_df = data_dict['olist_order_items_df']
    olist_order_payments_df = data_dict['olist_order_payments_df']
    olist_order_reviews_df = data_dict['olist_order_reviews_df']
    olist_orders_df = data_dict['olist_orders_df']
    olist_products_df = data_dict['olist_products_df']
    olist_sellers_df = data_dict['olist_sellers_df']

    #----------------- Merge Datasets
    df = merge_all_datasets(olist_customers_df, olist_geolocation_df, olist_order_items_df, olist_order_payments_df,
                             olist_order_reviews_df, olist_orders_df, olist_products_df, olist_sellers_df)
    print("1 ------ Merged Datasets")

    # ----------------- Handle Missing Values
    df = handle_missing_values(df)
    print("2 ------ Handled Missing Values")
    
    #----------------- Preprocess Data 
    df = preprocessing(df, state_to_region)
    print("3 ------ Preprocessed Data")

    #----------------- Keep columns of interest
    df_final = df[['order_id', 'customer_id', 'customer_unique_id', 'seller_id', 'payment_value', 'Rating', 'region', 
                         'Product_weight_kg', 'distance_km', 'Product_category', 'Product_size', 'No_photos',
                         'Product_price', 'month', 'is_delivery_late', 'freight_value', 'product_category_name_encoded', 
                         'late_delivery_in_days', 'order_purchase_timestamp', 'order_delivered_customer_date', 
                         'order_estimated_delivery_date', 'review_comment_message']]
    
    #----------------- Save Clean Data
    df_final.to_csv("../../data/processed/data.csv", index=False)
    end = time.time()
    print(f"4 -------- Clean data saved! time: {end-start:.2f}")


if __name__ == "__main__":
    main()







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




# Helper dictionaries for weather features



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
    

#---- dealing missing values (partial fix)

def train_validate_model(df, missing_val_index_lst):

    features = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'seller_id_encoded', 'month']
    target = 'product_category_encoded'
    df.dropna(subset=features + [target], inplace=True)
    missing_val_index_lst = df[df['product_category_name'].isna()].index

    # dataset without missing values; this will be used to train model
    df_clean = df[~df.index.isin(missing_val_index_lst)]

    # Features and Target
    X, y = df_clean[features], df_clean[target]

    # Splitting into train & validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    # Testing model
    rf_model = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)
    print("Random Forest accuracy on training set: ", accuracy_score(y_val, y_pred))

    # Retraining on full model
    rf_model_full = RandomForestClassifier().fit(X, y)
    impute_data = df.loc[missing_val_index_lst][features]
    predicted_product_category = rf_model_full.predict(impute_data)
    
    return predicted_product_category, df


def impute_product_category(df):

    df_copy = df.copy()

    # Encode the 'Product_category' column
    le = LabelEncoder()
    df_copy['product_category_encoded'] = le.fit_transform(df_copy['product_category_name'])
    
    # Encode the 'Product_category' column
    le2 = LabelEncoder()
    df_copy['seller_id_encoded'] = le2.fit_transform(df_copy['seller_id'])

    # Index of missing values for product_category
    product_category_null_values_index = df_copy[df_copy['product_category_name'].isna()].index

    # Splitting into train & validation
    predicted_product_category, df = train_validate_model(df_copy, product_category_null_values_index)

    # Imputing Values
    df.loc[df['product_category_name'].isna(), 'product_category_encoded'] = predicted_product_category
    df['product_category_name'] = le.inverse_transform(df['product_category_encoded'])
    print(f"{len(product_category_null_values_index)} have been imputed")

    return df


def handle_missing_values(df):
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['month'] = df['order_purchase_timestamp'].dt.month
    df = df[df['order_status'].isin(['delivered', 'shipped'])]

    # drop 1 row from freight_value
    df.dropna(subset=['freight_value'], inplace=True)

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
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['region'] = df['customer_state'].map(state_to_region)  
    df['Product_weight_kg'] = df['product_weight_g']/1000
    df['Product_category'] = df['product_category_name']
    le  = LabelEncoder()
    df['Product_category_encoded'] = le.fit_transform(df['Product_category']) 
    df['Product_size'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df['No_photos'] = df['product_photos_qty']
    df['Product_price'] = df['price']
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['late_delivery_in_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_delivery_late'] = np.where(df['late_delivery_in_days'] > 0, 1, 0)
    df['Rating']= df['review_score']
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['season'] = df['order_purchase_timestamp'].dt.month
    df['distance_km'] = df.apply(lambda row: haversine(row['geolocation_lat_x'], row['geolocation_lng_x'], row['geolocation_lat_y'], row['geolocation_lng_y']), axis=1)
    df.dropna(subset=['distance_km'], inplace=True)
    #df['customer_experience'] = df['customer_experience'].fillna(df['review_score'])
    #df['seller_avg_rating'] = df['seller_avg_rating'].fillna(df['review_score'])    
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
    product_category_name_translation_df = data_dict['product_category_name_translation_df']
    olist_closed_deals_df = data_dict['olist_closed_deals_df']
    olist_marketing_qualified_leads_df = data_dict['olist_marketing_qualified_leads_df']

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
                         'Product_price', 'season', 'is_delivery_late', 'Product_price', 'freight_value', 
                         'Product_category_encoded', 'late_delivery_in_days', 'order_purchase_timestamp',
                         'order_delivered_customer_date', 'order_estimated_delivery_date']]
    
    #----------------- Save Clean Data
    df_final.to_csv("../../data/processed/data.csv", index=False)
    end = time.time()
    print(f"4 -------- Clean data saved! time: {end-start:.2f}")


if __name__ == "__main__":
    main()







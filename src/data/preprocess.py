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
import time
from sklearn.impute import KNNImputer
from make_dataset import load_data, merge_all_datasets

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings("ignore")

state_to_region = {
    'AC': 'North', 'AP': 'North', 'AM': 'North', 'PA': 'North', 'RO': 'North', 'RR': 'North', 'TO': 'North',
    'AL': 'Northeast', 'BA': 'Northeast', 'CE': 'Northeast', 'MA': 'Northeast', 'PB': 'Northeast', 'PE': 'Northeast', 'PI': 'Northeast', 'RN': 'Northeast', 'SE': 'Northeast',
    'GO': 'Central-West', 'MT': 'Central-West', 'MS': 'Central-West', 'DF': 'Central-West',
    'ES': 'Southeast', 'MG': 'Southeast', 'RJ': 'Southeast', 'SP': 'Southeast',
    'PR': 'South', 'RS': 'South', 'SC': 'South'
}

class HandleMissingValues:
    '''This class  handles missing values by dropping or imputing through different functions tailored to each type of variable '''

    def dropping_values (self, df):
        '''Drops missing values since there is no logic way (Assumption_1) to impute them.'''
        df.dropna(subset=['seller_id','order_delivered_customer_date','review_score', 'freight_value'], inplace=True)
        return df 
    
    def geolocation_imputation (self, df):
        '''This code fills missing values in latitude and longitude columns of a DataFrame using the K-Nearest Neighbors'''
        geolocation_columns_x = ['geolocation_lat_x', 'geolocation_lng_x']
        geolocation_columns_y = ['geolocation_lat_y', 'geolocation_lng_y']
        knn_imputer = KNNImputer(n_neighbors=5)
        df[geolocation_columns_x] = knn_imputer.fit_transform(df[geolocation_columns_x])
        df[geolocation_columns_y] = knn_imputer.fit_transform(df[geolocation_columns_y])
        return df
    
    #----- product_category
    def get_features_and_target(self):
        '''Gets features and target for RF model'''
        features = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'seller_id_encoded', 'month']
        target = 'product_category_name_encoded'
        return features, target

    def encode_column(self, df, column):
        '''Encode Column in dataframe'''
        df2 = df.copy()
        le = LabelEncoder()
        df2[f'{column}_encoded'] = le.fit_transform(df2[column])
        return df2, le

    def train_random_forest_model(self, X, y):
        '''Fits Random Forest model'''
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
        rf1 = RandomForestClassifier().fit(X_train, y_train)
        y_pred = rf1.predict(X_val)
        print("Random Forest accuracy on training set: ", accuracy_score(y_val, y_pred)) 
        rf2 = RandomForestClassifier().fit(X, y)
        return rf2

    def impute_product_category(self, df):
        '''Imputes Product Category column'''
        df['month'] = pd.to_datetime(df['order_purchase_timestamp']).dt.month
        features, target = self.get_features_and_target()
        df, le = self.encode_column(df, 'product_category_name')
        df, le2 = self.encode_column(df, 'seller_id')
        df.dropna(subset=features + [target], inplace=True)

        # Model
        missing_val_index_lst = df[df['product_category_name'].isna()].index
        impute_data = df.loc[missing_val_index_lst][features] 
        df_clean = df[~df.index.isin(missing_val_index_lst)]
        X, y = df_clean[features], df_clean[target]
        rf_model_full = self.train_random_forest_model(X, y)
        imputed_product_category = rf_model_full.predict(impute_data)

        df.loc[missing_val_index_lst, 'product_category_name_encoded'] = imputed_product_category
        df['product_category_name'] = le.inverse_transform(df['product_category_name_encoded'])
        print(f"{len(missing_val_index_lst)} have been imputed")
        return df
   
    def dropping_columns (self, df):
        '''This function drops columns not used in analysis'''
        columns_to_drop = ['geolocation_zip_code_prefix_x', 'geolocation_zip_code_prefix_y', 'geolocation_city_x',
                           'geolocation_state_x', 'geolocation_city_y', 'geolocation_state_y', 'product_name_lenght',
                           'shipping_limit_date', 'payment_sequential', 'product_description_lenght', 'review_comment_title',
                           'order_delivered_carrier_date', 'payment_type', 'payment_installments', 'review_id', 'review_comment_message']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        return df

class ProcessingHelpers:
    '''This class hosts all helper functions needed for further processing.'''

    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  # Radius of Earth in kilometers
        return r * c

    def rolling_mean_customer(self, group):
        """Calculate expanding mean for customer experience."""
        filtered_reviews = group[group <= 5]
        expanding_mean = filtered_reviews.expanding().mean()
        return expanding_mean.reindex(group.index).ffill().bfill()

    def rolling_mean_seller(self, group):
        """Calculate expanding mean for seller experience."""
        filtered_reviews = group[group <= 5]
        expanding_mean = filtered_reviews.expanding().mean()
        return expanding_mean.reindex(group.index).ffill().bfill()

class Preprocess:
    '''Class for preprocessing data, using previous helper and imputation methods.'''

    def preprocessing(self, df, state_to_region):
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')
        df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'], errors='coerce')
        df['rainfall'] = df['customer_state'].map(state_to_region)
        df['product_weight_kg'] = df['product_weight_g'] / 1000
        df['product_size'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
        df['no_photos'] = df['product_photos_qty']
        df['product_price'] = df['price']
        df['late_delivery_in_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
        df['is_delivery_late'] = np.where(df['late_delivery_in_days'] > 0, 1, 0)
        df['distance_km'] = df.apply(lambda row: ProcessingHelpers().haversine(
            row['geolocation_lat_x'], row['geolocation_lng_x'], 
            row['geolocation_lat_y'], row['geolocation_lng_y']), axis=1)
        df.dropna(subset=['distance_km'], inplace=True)
        return df

    def rolling_mean_process(self, df):
        df = df.sort_values(by=['customer_id', 'review_answer_timestamp'])
        df['customer_experience'] = df.groupby('customer_id')['review_score'].apply(ProcessingHelpers().rolling_mean_customer).reset_index(level=0, drop=True)
        df['customer_experience'].fillna(df['review_score'], inplace=True)
        df = df.sort_values(by=['seller_id', 'review_answer_timestamp'])
        df['seller_avg_rating'] = df.groupby('seller_id')['review_score'].apply(ProcessingHelpers().rolling_mean_seller).reset_index(level=0, drop=True)
        df['seller_avg_rating'].fillna(df['review_score'], inplace=True)
        return df

    def df_final(self, df):
        columns_of_interest = [
            'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp',
            'order_approved_at', 'review_answer_timestamp', 'order_item_id', 
            'product_id', 'seller_id', 'payment_value', 'review_score', 'month',
            'rainfall', 'product_weight_kg', 'product_size', 'no_photos', 
            'product_price', 'is_delivery_late', 'customer_experience', 
            'seller_avg_rating', 'freight_value', 'distance_km', 'product_category_name',
            'product_category_name_encoded']
        return df[columns_of_interest]
    

if __name__ == "__main__":
    # Loading Data
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

    df = merge_all_datasets(olist_customers_df, olist_geolocation_df, olist_order_items_df,
                            olist_order_payments_df, olist_order_reviews_df, olist_orders_df,
                            olist_products_df, olist_sellers_df)
    
    handle_missing = HandleMissingValues()
    preprocess_helper = Preprocess()

    # Run the methods
    df = handle_missing.dropping_values(df)
    df = handle_missing.geolocation_imputation(df)
    df = handle_missing.impute_product_category(df)
    df = handle_missing.dropping_columns(df)

    df = preprocess_helper.preprocessing(df, state_to_region)
    df = preprocess_helper.rolling_mean_process(df)
    df_final = preprocess_helper.df_final(df)
    df_final.rename(columns={'review_score': 'Rating', 'season': 'month'}, inplace=True)
    df_final.to_csv("../../data/processed/data.csv", index=False)
    print("clean data saved!!")





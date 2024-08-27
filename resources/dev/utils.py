import pandas as pd
import numpy as np

# ----------------------------------------------------- 1
def load_data():
    data_files = {
        'olist_customers_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_customers_dataset.csv',
        'olist_geolocation_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_geolocation_dataset.csv',
        'olist_order_items_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_order_items_dataset.csv',
        'olist_order_payments_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_order_payments_dataset.csv',
        'olist_order_reviews_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_order_reviews_dataset.csv',
        'olist_orders_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_orders_dataset.csv',
        'olist_products_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_products_dataset.csv',
        'olist_sellers_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/olist_sellers_dataset.csv',
        'product_category_name_translation_df': '/Users/juanherrera/Desktop/causal-attribution-analysis/data/product_category_name_translation.csv'
    }

    dataframes = {name: pd.read_csv(path) for name, path in data_files.items()}

    return dataframes


    
            


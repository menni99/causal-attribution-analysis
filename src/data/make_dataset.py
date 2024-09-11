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
from pathlib import Path


# ----------------------------------------------------- 1
# def load_data() -> dict:
#     '''
#     ToDo: 
#         1. Write docstring
#         2. host data somwhere so we can pull from a centralized place
#     '''
#     data_files = {
#         'olist_customers_df': 'causal-attribution-analysis/data/raw/olist_customers_dataset.csv',
#         'olist_geolocation_df': 'causal-attribution-analysis/data/raw/olist_geolocation_dataset.csv',
#         'olist_order_items_df': 'causal-attribution-analysis/data/raw/olist_order_items_dataset.csv',
#         'olist_order_payments_df': 'causal-attribution-analysis/data/raw/olist_order_payments_dataset.csv',
#         'olist_order_reviews_df': 'causal-attribution-analysis/data/raw/olist_order_reviews_dataset.csv',
#         'olist_orders_df': 'causal-attribution-analysis/data/raw/olist_orders_dataset.csv',
#         'olist_products_df': 'causal-attribution-analysis/data/raw/olist_products_dataset.csv',
#         'olist_sellers_df': 'causal-attribution-analysis/data/raw/olist_sellers_dataset.csv',
#         'product_category_name_translation_df': 'causal-attribution-analysis/data/raw/product_category_name_translation.csv',
        
#         # Marketing data
#         'olist_closed_deals_df': 'causal-attribution-analysis/data/raw/marketing_data/olist_closed_deals_dataset.csv',
#         'olist_marketing_qualified_leads_df': 'causal-attribution-analysis/data/raw/marketing_data/olist_marketing_qualified_leads_dataset.csv'

#     }

#     dataframes = {name: pd.read_csv(path) for name, path in data_files.items()}
#     return dataframes




def load_data() -> dict:
    '''
    Load raw data from CSV files located in the data/raw directory.

    Returns:
        A dictionary containing dataframes for all required datasets.
    '''
    # Define the base directory relative to the current script's location
    base_dir = Path(__file__).resolve().parent.parent.parent  # Adjust based on your project structure

    data_files = {
        'olist_customers_df': base_dir / 'data/raw/olist_customers_dataset.csv',
        'olist_geolocation_df': base_dir / 'data/raw/olist_geolocation_dataset.csv',
        'olist_order_items_df': base_dir / 'data/raw/olist_order_items_dataset.csv',
        'olist_order_payments_df': base_dir / 'data/raw/olist_order_payments_dataset.csv',
        'olist_order_reviews_df': base_dir / 'data/raw/olist_order_reviews_dataset.csv',
        'olist_orders_df': base_dir / 'data/raw/olist_orders_dataset.csv',
        'olist_products_df': base_dir / 'data/raw/olist_products_dataset.csv',
        'olist_sellers_df': base_dir / 'data/raw/olist_sellers_dataset.csv',
        'product_category_name_translation_df': base_dir / 'data/raw/product_category_name_translation.csv',
        
        # Marketing data
        'olist_closed_deals_df': base_dir / 'data/raw/marketing_data/olist_closed_deals_dataset.csv',
        'olist_marketing_qualified_leads_df': base_dir / 'data/raw/marketing_data/olist_marketing_qualified_leads_dataset.csv'
    }

    dataframes = {name: pd.read_csv(path) for name, path in data_files.items()}
    return dataframes


base_dir = Path(__file__).resolve().parent.parent.parent
print(base_dir)


# ----------------------------------------------------- 2
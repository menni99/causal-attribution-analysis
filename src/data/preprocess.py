import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



# ------------
# ------------------------------------------------------------- Define the Haversine function
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


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def rolling_mean_excluding_last(group):
    return group['review_score'].shift(1).expanding().mean()

def preprocessing(df, state_to_region):
    # Convert 'order_approved_at' column to datetime if it's not already
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])

    # Extract the month from 'order_approved_at'
    df['month'] = df['order_approved_at'].dt.month

    # Map state to region
    df['rainfall'] = df['customer_state'].map(state_to_region)  

    # Weight
    df['Product_weight_kg'] = df['product_weight_g'] / 1000

    # Product category
    df['Product_category'] = df['product_category_name']
    le = LabelEncoder()
    df['Product_category_encoded'] = le.fit_transform(df['Product_category']) 

    # Product size
    df['Product_size'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']

    # Number of photos
    df['No_photos'] = df['product_photos_qty']

    # Product price
    df['Product_price'] = df['price']

    # Date conversions
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    df['late_delivery_in_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_delivery_late'] = np.where(df['late_delivery_in_days'] > 0, 1, 0)

    # Rating
    df['Rating'] = df['review_score']

    # Seasonality
    df['seasonality'] = df['order_purchase_timestamp'].dt.month

    # Calculate distance_km
    df['distance_km'] = df.apply(lambda row: haversine(row['geolocation_lat_x'], row['geolocation_lng_x'],
                                                      row['geolocation_lat_y'], row['geolocation_lng_y']), axis=1)

    # Sort by customer_id for customer_experience calculation
    df = df.sort_values(by=['customer_id', 'review_answer_timestamp'])

    # Create a new column for the rolling mean of previous product ratings for customer experience
    df['customer_experience'] = df.groupby('customer_id').apply(rolling_mean_excluding_last).reset_index(drop=True)
    df['customer_experience'] = df['customer_experience'].fillna(df['review_score'])

    # Sort by seller_id for seller_avg_rating calculation
    df = df.sort_values(by=['seller_id', 'review_answer_timestamp'])

    # Create a new column for the rolling mean of previous product ratings for seller average rating
    df['seller_avg_rating'] = df.groupby('seller_id').apply(rolling_mean_excluding_last).reset_index(drop=True)
    df['seller_avg_rating'] = df['seller_avg_rating'].fillna(df['review_score'])

    # rain_level
    def get_rainfall_category(row):
    if pd.isnull(row['seller_state']) or pd.isnull(row['order_purchase_timestamp']):
        return 'Unknown'

    month = pd.to_datetime(row['order_purchase_timestamp']).month
    region = state_to_region.get(row['seller_state'])
    
    if region:
        return rainfall_categories[region].get(month, 'Unknown')
    return 'Unknown'
    df['rain_level'] = df.apply(get_rainfall_category, axis=1)

    # Create final DataFrame with selected columns
    df_final = df[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 
                   'review_answer_timestamp', 'order_item_id', 'product_id', 'seller_id', 'payment_value', 
                   'review_id', 'review_score', 'month', 'rainfall', 'Product_weight_kg', 'Product_category', 
                   'Product_size', 'No_photos', 'Product_price', 'seasonality', 'is_delivery_late', 
                   'geolocation_lat_x', 'geolocation_lng_x', 'geolocation_lat_y', 'geolocation_lng_y', 
                   'freight_value', 'distance_km', 'Product_category_encoded', 'customer_experience', 'seller_avg_rating']]
    
    return df_final



# ------------------ dealing missing values

import pandas as pd
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode


class FeatureEngineering:
    def __init__(self, threshold=10, columns_to_scale=None):
        """
        Initialize the feature engineering process with configuration options.

        Parameters:
        - threshold: The minimum number of occurrences to consider a category in high-cardinality features (default: 10).
        - columns_to_scale: List of columns to scale (default: None, which scales a predefined set of columns).
        """
        self.threshold = threshold
        self.columns_to_scale = columns_to_scale if columns_to_scale else [
            'base_price', 'initial_quantity', 'sold_quantity', 'city_name'
        ]

    def filter_relevant_features(self, data):
        """Filter only the relevant columns for the analysis."""
        final_variables = [
            'base_price', 'listing_type_id', 'warranty', 'tags', 'initial_quantity', 
            'sold_quantity', 'start_time', 'last_updated', 'shipping', 'seller_address',
            'automatic_relist', 'non_mercado_pago_payment_methods'
        ]
        return data[final_variables]

    def get_dummies(self, data):
        """Convert categorical columns to dummy variables."""
        print('Getting dummies...')
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        return pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    def transform_categorical_data_train_test(self, train_data, test_data, column_name):
        """Handle high-cardinality categorical variables."""
        column_counts = train_data[column_name].value_counts()
        valid_categories = column_counts[column_counts >= self.threshold].index
        
        for df in [train_data, test_data]:
            df[column_name] = df[column_name].apply(
                lambda x: x if x in valid_categories else 'OTHER'
            )
        return train_data, test_data

    def normalize_data(self, train_data, test_data=None, scaler=None):
        """Normalize specified columns using StandardScaler."""
        if scaler is None:
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_data[self.columns_to_scale])
            test_scaled = scaler.transform(test_data[self.columns_to_scale]) if test_data is not None else None
        else:
            train_scaled = scaler.transform(train_data[self.columns_to_scale])
            test_scaled = scaler.transform(test_data[self.columns_to_scale]) if test_data is not None else None

        train_data = self._replace_scaled_columns(train_data, train_scaled)
        if test_data is not None:
            test_data = self._replace_scaled_columns(test_data, test_scaled)

        return train_data, test_data, scaler

    def _replace_scaled_columns(self, data, scaled_data):
        """Replace scaled columns in the original dataframe."""
        scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in self.columns_to_scale], index=data.index)
        return pd.concat([data.drop(self.columns_to_scale, axis=1), scaled_df], axis=1)

    def transform_warranty(self, value):
        """Transform the warranty column."""
        if value is None or value == "":
            return "no info"
        value = str(value).lower()
        value = unidecode(value)
        if value in ["si", "sí", "s", "yes"] or "mes" in value or "ano" in value or "año" in value:
            return "con garantia"
        if any(keyword in value for keyword in ["sin garantia", "no", "ninguna"]):
            return "sin garantia"
        if "garantia" in value or "garantizamos" in value:
            return "otros con garantia"
        return "no info"

    def transform_tags(self, data):
        """Transform and expand 'tags' into dummy variables."""
        print('Transforming tags...')
        tags_dummies = data['tags'].explode().str.get_dummies().groupby(level=0).sum()
        return pd.concat([data.drop(columns=['tags']), tags_dummies], axis=1)
    
    def transform_dates(self, data):
        print('Transforming dates...')
        """Transform and extract features from date columns."""
        data['start_time'] = pd.to_datetime(data['start_time'], utc=True).dt.tz_localize(None)
        data['last_updated'] = pd.to_datetime(data['last_updated'], utc=True).dt.tz_localize(None)
        data['updated_since_creation'] = (data['last_updated'] - data['start_time']).dt.days
        data['updated_label'] = (data['updated_since_creation'] >= 1).astype(int)
        return data.drop(columns=['start_time', 'last_updated'])

    def transform_shipping(self, data):
        """Extract and transform shipping information."""
        print('Transforming shipping...')
        data['local_pick_up'] = data['shipping'].apply(lambda x: x.get('local_pick_up') if isinstance(x, dict) else None)
        data['free_shipping'] = data['shipping'].apply(lambda x: x.get('free_shipping') if isinstance(x, dict) else None)
        data['local_pick_up'] = data['local_pick_up'].astype(int)
        data['free_shipping'] = data['free_shipping'].astype(int)
        return data.drop(columns='shipping')

    def transform_automatic_relist(self, data):
        """Transform the automatic relist feature."""
        print('Transforming automatic relist')
        data['automatic_relist'] = data['automatic_relist'].fillna(False).astype(int)
        return data

    def transform_address(self, data):
        """Standardize and transform address information."""
        print('Transforming address...')
        def clean_text(value):
            value = str(value).lower()
            value = unidecode(value)
            value = ''.join(e for e in value if e.isalnum() or e.isspace())
            return value.strip()
        data['city_name'] = data['seller_address'].apply(
            lambda x: x['city']['name'] if isinstance(x, dict) and 'city' in x else None
        )
        data['city_name'] = data['city_name'].apply(clean_text)
        data['city_name'] = data['city_name'].replace(['capital federal', 'caba', 'buenos aires',
                                                        'ciudad autonoma de buenos aires'],
                                                        'buenos aires')
        city_counts = data['city_name'].value_counts()
        data['city_name'] = data['city_name'].map(city_counts)
        return data.drop(columns=['seller_address'])

    def transform_payment(self, data):
        """Transform payment method data."""
        print('Transforming payment')
        def normalize_text(text):
            return unidecode(text.lower())

        def create_columns_payment(data):
            categories = {
                'credit_card': ['visa', 'mastercard', 'american express', 'visa electron', 'mastercard maestro', 'diners'],
                'bank_transfer': ['transferencia bancaria'],
                'cash': ['efectivo'],
                'marketplace_payment': ['mercadopago'],
                'cheque': ['cheque certificado'],
                'other': ['acordar con el comprador', 'giro postal']
            }

            payment_categories = {category: 0 for category in categories.keys()}

            for method in data:
                description = normalize_text(method['description'])
                matched = False

                for category, values in categories.items():
                    if description in values:
                        payment_categories[category] = 1
                        matched = True
                        break

                if not matched:
                    payment_categories['other'] = 1

            return payment_categories

        payment_columns = data['non_mercado_pago_payment_methods'].apply(create_columns_payment)
        df_payment_methods = pd.json_normalize(payment_columns)
        df_payment_methods.index = data.index
        return pd.concat([data, df_payment_methods], axis=1).drop(columns='non_mercado_pago_payment_methods')

    def feature_engineering(self, data):
        """Apply all feature engineering transformations."""
        data = self.filter_relevant_features(data)
        data['warranty'] = data['warranty'].apply(self.transform_warranty)
        data = self.transform_automatic_relist(data)
        data = self.transform_tags(data)
        data = self.transform_dates(data)
        data = self.transform_shipping(data)
        data = self.transform_address(data)
        data = self.transform_payment(data)
        data = self.get_dummies(data)
        return data


# Guard clause to prevent execution when imported
if __name__ == "__main__":
    print("This module is meant to be imported into a main.py file.")

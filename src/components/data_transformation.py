import sys
from dataclasses import dataclass
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import os
pd.set_option('future.no_silent_downcasting', True)



@dataclass
class DataTransformationConfig:
    data_path: str = os.path.join('data', "clean_data_after_eda.csv")
    price_data_path: str = os.path.join('data', "price_data.csv")
    transformed_data_path: str = os.path.join('data', "transformed_data.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            logging.info("Initiated data transformation")

            df = pd.read_csv(self.data_transformation_config.data_path)

            logging.info("Loaded the data")

            df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
            df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
            df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
            df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')

            # Calculate Difference between off-peak prices in December and preceding January
            price_df = pd.read_csv(self.data_transformation_config.price_data_path)
            price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')
            # Group off-peak prices by companies and month
            monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg(
                {'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

            # Get january and december prices
            jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
            dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

            # Calculate the difference
            diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}),
                            jan_prices.drop(columns='price_date'), on='id')
            diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
            diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
            diff = diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]

            df = pd.merge(df, diff, on='id')
            print(df.head())

            # Aggregate average prices per period by company
            mean_prices = price_df.groupby(['id']).agg({
                'price_off_peak_var': 'mean',
                'price_peak_var': 'mean',
                'price_mid_peak_var': 'mean',
                'price_off_peak_fix': 'mean',
                'price_peak_fix': 'mean',
                'price_mid_peak_fix': 'mean'
            }).reset_index()

            # Calculate the mean difference between consecutive periods
            mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices[
                'price_peak_var']
            mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices[
                'price_mid_peak_var']
            mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices[
                'price_mid_peak_var']
            mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices[
                'price_peak_fix']
            mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices[
                'price_mid_peak_fix']
            mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices[
                'price_mid_peak_fix']

            columns = [
                'id',
                'off_peak_peak_var_mean_diff',
                'peak_mid_peak_var_mean_diff',
                'off_peak_mid_peak_var_mean_diff',
                'off_peak_peak_fix_mean_diff',
                'peak_mid_peak_fix_mean_diff',
                'off_peak_mid_peak_fix_mean_diff'
            ]
            df = pd.merge(df, mean_prices[columns], on='id')

            # Aggregate average prices per period by company
            mean_prices_by_month = price_df.groupby(['id', 'price_date']).agg({
                'price_off_peak_var': 'mean',
                'price_peak_var': 'mean',
                'price_mid_peak_var': 'mean',
                'price_off_peak_fix': 'mean',
                'price_peak_fix': 'mean',
                'price_mid_peak_fix': 'mean'
            }).reset_index()

            # Calculate the mean difference between consecutive periods
            mean_prices_by_month['off_peak_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - \
                                                                  mean_prices_by_month['price_peak_var']
            mean_prices_by_month['peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_peak_var'] - \
                                                                  mean_prices_by_month['price_mid_peak_var']
            mean_prices_by_month['off_peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - \
                                                                      mean_prices_by_month['price_mid_peak_var']
            mean_prices_by_month['off_peak_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - \
                                                                  mean_prices_by_month['price_peak_fix']
            mean_prices_by_month['peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_peak_fix'] - \
                                                                  mean_prices_by_month['price_mid_peak_fix']
            mean_prices_by_month['off_peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - \
                                                                      mean_prices_by_month['price_mid_peak_fix']

            # Calculate the maximum monthly difference across time periods
            max_diff_across_periods_months = mean_prices_by_month.groupby(['id']).agg({
                'off_peak_peak_var_mean_diff': 'max',
                'peak_mid_peak_var_mean_diff': 'max',
                'off_peak_mid_peak_var_mean_diff': 'max',
                'off_peak_peak_fix_mean_diff': 'max',
                'peak_mid_peak_fix_mean_diff': 'max',
                'off_peak_mid_peak_fix_mean_diff': 'max'
            }).reset_index().rename(
                columns={
                    'off_peak_peak_var_mean_diff': 'off_peak_peak_var_max_monthly_diff',
                    'peak_mid_peak_var_mean_diff': 'peak_mid_peak_var_max_monthly_diff',
                    'off_peak_mid_peak_var_mean_diff': 'off_peak_mid_peak_var_max_monthly_diff',
                    'off_peak_peak_fix_mean_diff': 'off_peak_peak_fix_max_monthly_diff',
                    'peak_mid_peak_fix_mean_diff': 'peak_mid_peak_fix_max_monthly_diff',
                    'off_peak_mid_peak_fix_mean_diff': 'off_peak_mid_peak_fix_max_monthly_diff'
                }
            )

            columns = [
                'id',
                'off_peak_peak_var_max_monthly_diff',
                'peak_mid_peak_var_max_monthly_diff',
                'off_peak_mid_peak_var_max_monthly_diff',
                'off_peak_peak_fix_max_monthly_diff',
                'peak_mid_peak_fix_max_monthly_diff',
                'off_peak_mid_peak_fix_max_monthly_diff'
            ]

            df = pd.merge(df, max_diff_across_periods_months[columns], on='id')
            df.head()

            # Transforming Boolean data
            df['has_gas'] = df['has_gas'].replace(['t', 'f'], [1, 0])
            df.groupby(['has_gas']).agg({'churn': 'mean'})

            # Transform into categorical type
            df['channel_sales'] = df['channel_sales'].astype('category')
            # Let's see how many categories are within this column
            df['channel_sales'].value_counts()
            df = pd.get_dummies(df, columns=['channel_sales'], prefix='channel')
            df = df.drop(
                columns=['channel_sddiedcslfslkckwlfkdpoeeailfpeds', 'channel_epumfxlbckeskwekxbiuasklxalciiuu',
                         'channel_fixdbufsefwooaasfcxdxadsiekoceaa'])
            df.head()

            # Transforming skewed data The reason why we need to treat skewness is because some predictive models
            # have inherent assumptions about the distribution of the features that are being supplied to it. Such
            # models are called parametric models, and they typically assume that all variables are both independent
            # and normally distributed. Skewness isn't always a bad thing, but as a rule of thumb it is always good
            # practice to treat highly skewed variables because of the reason stated above, but also as it can
            # improve the speed at which predictive models are able to converge to its best solution.

            skewed = [
                'cons_12m',
                'cons_gas_12m',
                'cons_last_month',
                'forecast_cons_12m',
                'forecast_cons_year',
                'forecast_discount_energy',
                'forecast_meter_rent_12m',
                'forecast_price_energy_off_peak',
                'forecast_price_energy_peak',
                'forecast_price_pow_off_peak'
            ]

            df[skewed].describe()

            # Apply log10 transformation
            df["cons_12m"] = np.log10(df["cons_12m"] + 1)
            df["cons_gas_12m"] = np.log10(df["cons_gas_12m"] + 1)
            df["cons_last_month"] = np.log10(df["cons_last_month"] + 1)
            df["forecast_cons_12m"] = np.log10(df["forecast_cons_12m"] + 1)
            df["forecast_cons_year"] = np.log10(df["forecast_cons_year"] + 1)
            df["forecast_meter_rent_12m"] = np.log10(df["forecast_meter_rent_12m"] + 1)
            df["imp_cons"] = np.log10(df["imp_cons"] + 1)

            fig, axs = plt.subplots(nrows=3, figsize=(18, 20))
            # Plot histograms
            sns.histplot((df["cons_12m"].dropna()), ax=axs[0])
            sns.histplot((df[df["has_gas"] == 1]["cons_gas_12m"].dropna()), ax=axs[1])
            sns.histplot((df["cons_last_month"].dropna()), ax=axs[2])

            plt.savefig(os.path.join('images', "skew_transformed.png"))

            plt.show()


            df.head()

            df.to_csv(self.data_transformation_config.transformed_data_path)

            logging.info("Exited data transformation")


        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    obj.data_transformer()

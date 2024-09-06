import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from src.utils import plot_stacked_bars, plot_distribution
sns.set(color_codes=True)

@dataclass
class DataAnalysisConfig:
    client_data_path: str = os.path.join('data', "client_data.csv")
    price_data_path: str = os.path.join('data', "price_data.csv")


class DataAnalysis:
    def __init__(self):
        self.analysis_config = DataAnalysisConfig()

    def initiate_data_analysis(self):
        logging.info("Entered the data analysis component")
        try:
            client_df = pd.read_csv(self.analysis_config.client_data_path)
            price_df = pd.read_csv(self.analysis_config.price_data_path)
            logging.info('Read the dataset as dataframe')

            print(client_df.info())
            print(price_df.info())

            churn = client_df[['id', 'churn']]
            churn.columns = ['Companies', 'churn']
            churn_total = churn.groupby(churn['churn']).count()
            churn_percentage = churn_total / churn_total.sum() * 100

            # plot to visualize the churn percentage
            plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")

            channel = client_df[['id', 'channel_sales', 'churn']]
            channel = channel.groupby([channel['channel_sales'], channel['churn']])['id'].count().unstack(
                level=1).fillna(0)
            channel_churn = (channel.div(channel.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)

            # Channel wise churn rate analysis
            plot_stacked_bars(channel_churn, 'Sales channel', rot_=30)

            # Histograms to visualize distribution of the consumption in the last year and month
            consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

            fig, axs = plt.subplots(nrows=4, figsize=(18, 25))

            plot_distribution(consumption, 'cons_12m', axs[0])
            plot_distribution(consumption[consumption['has_gas'] == 't'], 'cons_gas_12m', axs[1])
            plot_distribution(consumption, 'cons_last_month', axs[2])
            plot_distribution(consumption, 'imp_cons', axs[3])

            plt.savefig(os.path.join('images', "consumption.png"))
            plt.show()

            # Clearly, the consumption data is highly positively skewed, presenting a very long right-tail
            # towards the higher values of the distribution. The values on the higher and lower end of the
            # distribution are likely to be outliers. We can use a standard plot to visualise the outliers
            # in more detail.

            # A boxplot is a standardized way of displaying the distribution to reveal skewness
            fig, axs = plt.subplots(nrows=4, figsize=(18, 25))

            # Plot histogram
            sns.boxplot(consumption["cons_12m"], ax=axs[0])
            sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], ax=axs[1])
            sns.boxplot(consumption["cons_last_month"], ax=axs[2])
            sns.boxplot(consumption["imp_cons"], ax=axs[3])
            plt.savefig(os.path.join('images', "box_plot.png"))
            plt.show()


            contract_type = client_df[['id', 'has_gas', 'churn']]
            contract = contract_type.groupby([contract_type['churn'], contract_type['has_gas']])['id'].count().unstack(
                level=0)
            contract_percentage = (contract.div(contract.sum(axis=1), axis=0) * 100).sort_values(by=[1],
                                                                                                 ascending=False)
            # Churn rate based on contract types
            plot_stacked_bars(contract_percentage, 'Contract type (with gas')

            logging.info("Exploratory data analysis completed")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataAnalysis()
    obj.initiate_data_analysis()
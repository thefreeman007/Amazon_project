import warnings
warnings.filterwarnings('ignore')
import os
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pickle
import keepa
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from tqdm import tqdm
from prophet.make_holidays import make_holidays_df
import traceback

# 日志配置
# Logging configuration
logging.basicConfig(filename="process_log.txt", level=logging.INFO)

# Pandas 数据显示设置
# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 6000)

# 全局路径变量，用于管理数据的输入输出
# Global path variables for managing data input and output
DATA_DICT_PATH = "../data/all_dict/"
DATA_DF_PATH = "../data/all_df/"
OUTPUT_PATH = "../output/"
# 产品类型关键词，用于筛选文件
# Product type keyword for filtering files
TYPE_KEYWORD="Health"
# 测试集长度
# Length of test set
test_num = 60


def get_dict_list():
    """
    获取字典文件路径列表
    Retrieve the list of dictionary file paths
    """
    return [os.path.join(DATA_DICT_PATH, f) for f in listdir(DATA_DICT_PATH)]


def get_predict_data_list():
    """
    获取预测数据文件路径列表
    Retrieve the list of prediction data file paths
    """
    return [os.path.join(DATA_DF_PATH, f) for f in listdir(DATA_DF_PATH)]


def Organize_raw_data_and_create_CSV():
    """
    整理字典数据并生成 CSV 文件。
    Process dictionary data and generate CSV files.

    功能：
    - 解析字典文件中的数据，验证数据的完整性。
    - 对销量数据进行时间聚合，并填补缺失值。
    - 保存处理后的 CSV 文件。

    Functionality:
    - Parse data from dictionary files and validate its integrity.
    - Aggregate sales data over time and fill missing values.
    - Save the processed data as CSV files.
    """
    csv_list = get_dict_list()
    df_completed_list = pd.DataFrame()  # 保存已处理的文件列表 / Save the list of processed files

    for i in csv_list:
        print(f'Processing product code: {i}')
        dictfile = open(i, 'rb')  # 打开二进制文件 / Open the binary file
        try:
            products = pickle.load(dictfile)
            dictfile.close()
            products_csv = keepa.parse_csv(products[0]['csv'])  # 解析 CSV 数据 / Parse the CSV data

            # 检查销量数据是否满足要求：长度大于30 / Check if sales data meets requirements: length > 30
            if 'df_SALES' in products_csv.keys() and len(products_csv['df_SALES']) > 30:
                df_csv = products_csv['df_SALES']
                df_csv = pd.DataFrame(df_csv)

                # 删除错误数据并重置格式 / Remove erroneous data and reset format
                df_csv = df_csv[2:]
                df_csv.reset_index(inplace=True, drop=False)
                df_csv.rename(columns={'index': 'date'}, inplace=True)

                # 转换日期列格式为 datetime / Convert date column format to datetime
                df_csv['date'] = pd.to_datetime(df_csv['date'], errors='coerce')
                df_csv['date'] = df_csv['date'].dt.date

                # 按日期聚合数据并补充缺失天数 / Aggregate data by date and add missing days
                date_index = pd.date_range(start=df_csv['date'].min(), end=df_csv['date'].max(), freq='D')
                df_csv = df_csv.groupby('date')['value'].max().reset_index(drop=False)
                df_csv = df_csv.set_index('date').reindex(date_index)
                df_csv.reset_index(inplace=True, drop=False)
                df_csv.rename(columns={'index': 'date'}, inplace=True)
                df_csv['date'] = pd.to_datetime(df_csv['date'], errors='coerce')

                # 前向填充缺失值 / Forward-fill missing values
                df_csv['value'] = df_csv['value'].fillna(method='ffill')

                # 保存分类树或父 ASIN 信息到 CSV / Save category tree or parent ASIN information to CSV
                if products[0]['categoryTree'] is not None:
                    for g in range(len(products[0]['categoryTree'])):
                        if isinstance(products[0]['categoryTree'], list):
                            df_csv[f'name_{i[-10:]}'] = products[0]['categoryTree'][g]['name']
                            df_csv.to_csv(
                                f'{DATA_DF_PATH}{i[-10:]}_{len(df_csv)}_{products[0]["categoryTree"][0]["name"]}.csv')
                        elif products[0]['categoryTree'][i] is None:
                            df_csv.to_csv(f'{DATA_DF_PATH}{i[-10:]}_{len(df_csv)}.csv')
                        else:
                            df_csv[f'name_{i[-10:]}'] = products[0]['categoryTree'][g]['name']
                            df_csv.to_csv(
                                f'{DATA_DF_PATH}{i[-10:]}_{len(df_csv)}_{products[0]["categoryTree"][0]["name"]}.csv')
                else:
                    pos = len(df_completed_list)
                    df_completed_list.loc[pos, 'product'] = i
                    if isinstance(products[0]['parentAsin'], str):
                        df_completed_list.loc[pos, 'product_list_0'] = products[0]['parentAsin']

        except FileNotFoundError as fnf_error:
            # 捕获文件未找到的异常 / Handle FileNotFoundError
            print(f"Error processing product {i}: {fnf_error}")
            logging.error(f"File not found: {dictfile}, error: {fnf_error}")

        except ValueError as value_error:
            # 捕获值处理异常 / Handle ValueError
            logging.error(f"Value error in file: {dictfile}, error: {value_error}")

        except Exception as generic_error:
            # 捕获其他通用异常 / Handle generic exceptions
            logging.error(f"Unexpected error in file: {dictfile}, error: {generic_error}")
    df_completed_list.to_csv('df_completed_list.csv')  # 保存完成列表 / Save completed list


def check_file_type(filename, TYPE_KEYWORD):
    """
    检查文件名中是否包含特定的关键字。
    Check if the filename contains a specific keyword.

    参数:
    - filename (str): 文件名 / The name of the file.
    - TYPE_KEYWORD (str): 要查找的关键字 / The keyword to search for.

    返回:
    - bool: 如果包含关键字，则返回 True，否则返回 False / Returns True if the keyword is present, otherwise False.
    """
    return TYPE_KEYWORD in filename

def process_file(file_name, train=True, validate=True, predict=True):
    """
    处理单个文件并进行时间序列预测。
    Process a single file and perform time series forecasting.

    功能：
    - 从 CSV 文件加载数据，并分为训练集和测试集。
    - 使用 Prophet 模型进行时间序列预测。
    - 可选验证模型的预测性能。
    - 绘制预测结果与实际值对比图。

    Functionality:
    - Load data from a CSV file and split it into training and testing sets.
    - Perform time series forecasting using the Prophet model.
    - Optionally validate the model's forecasting performance.
    - Plot a comparison between forecasted and actual values.

    参数:
    - file_name (str): 文件名 / The name of the file.
    - train (bool): 是否训练模型 / Whether to train the model.
    - validate (bool): 是否验证模型 / Whether to validate the model.
    - predict (bool): 是否生成预测 / Whether to generate predictions.

    返回:
    - predict_results (DataFrame): 包含预测结果 / DataFrame with forecast results.
    - validate_results (DataFrame): 包含验证指标 / DataFrame with validation metrics.
    """
    try:
        predict_results = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'code'])
        validate_results = pd.DataFrame(columns=['product_code', 'mean_absolute_percentage_error', 'pred_coverage', 'Root_Mean_Squared_Error'])

        # 加载数据并检查数据长度 / Load data and check its length
        df = pd.read_csv(file_name, index_col=[0])
        if len(df) < 120:
            return predict_results, validate_results

        # 数据预处理 / Data preprocessing
        df = df[['date', 'value']]  # 选择需要的列 / Select required columns
        df = df[10:].reset_index(drop=True)
        df.columns = ['ds', 'y']  # 重命名为 Prophet 需要的列名 / Rename columns as required by Prophet

        # 数据分割为训练集和测试集 / Split data into training and testing sets
        df_train, df_test = df[:-test_num], df[-test_num:].reset_index(drop=True)

        # 定义节假日数据 / Define holiday data
        years = [2018, 2019, 2020, 2021, 2022]
        country = 'US'
        holidays = make_holidays_df(years, country)

        # 初始化 Prophet 模型 / Initialize the Prophet model
        model = Prophet(growth='linear', seasonality_mode='multiplicative', holidays=holidays)

        if train:
            # 训练 Prophet 模型 / Train the Prophet model
            model.fit(df_train)

            # 保存模型 / Save the model
            with open('prophet_model.pkl', 'wb') as f:
                pickle.dump(model, f)

        if predict:
            # 预测未来数据 / Predict future data
            future_df = model.make_future_dataframe(periods=len(df_test), freq='D')
            forecast = model.predict(future_df)

            # 提取预测结果 / Extract forecast results
            predict_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            predict_results['file_name'] = file_name  # 添加文件名 / Add filename

            # 合并预测与实际数据并绘图 / Merge forecast with actual data and plot
            df_test['ds'] = pd.to_datetime(df_test['ds'])
            comparison = pd.merge(df_test, predict_results, on='ds', how='inner')
            comparison['Actual_mean'] = comparison['y'].rolling(window=5).mean()

            plt.figure(figsize=(12, 6))
            plt.suptitle(TYPE_KEYWORD, fontsize=16)
            plt.plot(comparison['ds'], comparison['Actual_mean'], label='Actual_mean', color='red', linestyle='--')
            plt.plot(comparison['ds'], comparison['y'], label='Actual', color='blue', marker='o')
            plt.plot(comparison['ds'], comparison['yhat'], label='Predicted', color='orange', linestyle='--')
            plt.fill_between(comparison['ds'], comparison['yhat_lower'], comparison['yhat_upper'], color='orange', alpha=0.2, label='Prediction Interval')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('Actual vs Predicted')
            plt.legend()
            plt.grid()
            plt.show()

        if validate:
            # 使用 Prophet 交叉验证功能验证预测性能 / Validate forecasting performance using Prophet cross-validation
            train_num = len(df_train)
            df_cv = cross_validation(model, initial='%s days' % (train_num - 60), period='%s days' % 10, horizon='%s days' % 30)
            df_perf_metrics = performance_metrics(df_cv)

            validate_results = pd.DataFrame([{
                'product_code': file_name[15:25],
                'mean_absolute_percentage_error': df_perf_metrics['mape'].mean(),
                'pred_coverage': df_perf_metrics['coverage'].mean(),
                'Root_Mean_Squared_Error': df_perf_metrics['rmse'].mean(),
            }])

        return predict_results, validate_results

    except Exception as err:
        # 捕获并记录异常 / Catch and log exceptions
        print(err, traceback.format_exc())
        logging.error(f"Error processing file {file_name} : {err}")
        logging.error("Traceback:\n" + traceback.format_exc())

        # 返回空数据以确保程序继续运行 / Return empty data to ensure the program continues running
        return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'code']), pd.DataFrame(columns=['product_code', 'mean_absolute_percentage_error', 'pred_coverage', 'Root_Mean_Squared_Error'])

def main(train=False, validate=False, predict=False):
    """
    主程序入口，依赖整理后的 CSV 数据进行分析。
    Main entry point for the program, performs analysis based on preprocessed CSV data.

    功能：
    - 根据用户选择进行模型训练、验证和预测。
    - 将预测和验证结果保存为 CSV 文件。

    Functionality:
    - Perform model training, validation, and prediction based on user input.
    - Save forecast and validation results as CSV files.
    """
    data_list = [os.path.join(DATA_DF_PATH, f) for f in os.listdir(DATA_DF_PATH) if f.endswith('.csv') and TYPE_KEYWORD in f]

    df_predict = pd.DataFrame()
    df_validate = pd.DataFrame()

    for i in tqdm(range(len(data_list))):
        file = data_list[i]
        predict_results, validate_results = process_file(file_name=file, train=train, validate=validate, predict=predict)

        # 保存中间预测结果 / Save intermediate forecast results
        if predict_results is not None:
            df_predict = pd.concat([df_predict, predict_results], ignore_index=True)

        # 保存中间验证结果 / Save intermediate validation results
        if validate_results is not None:
            df_validate = pd.concat([df_validate, validate_results], ignore_index=True)

        # 定期保存结果 / Periodically save results
        if i % 10 == 0 or i == len(data_list) - 1:
            if predict:
                df_predict.to_csv(os.path.join(OUTPUT_PATH, 'predict_results.csv'), index=False)
            if validate:
                df_validate.to_csv(os.path.join(OUTPUT_PATH, 'validate_results.csv'), index=False)

    # 保存最终结果 / Save final results
    if predict:
        df_predict.dropna(inplace=True)
        df_predict.iloc[:, 1:] = df_predict.iloc[:, 1:].astype(float).round(3)
        df_predict.to_csv(os.path.join(OUTPUT_PATH, 'final_predict_results.csv'), index=False)
    if validate:
        df_validate.dropna(inplace=True)
        df_validate.iloc[:, 1:] = df_validate.iloc[:, 1:].astype(float).round(3)
        df_validate.to_csv(os.path.join(OUTPUT_PATH, 'final_validate_results.csv'), index=False)
    print("Processing completed and results saved.")

# 调用 make_csv 和 main
# Call make_csv and main
if __name__ == '__main__':
        # Step 1: 数据整理
        response = input("Step 1: Do you want to organize data? (y/n): ").strip().lower()
        train_input = input("Do you want to perform training? (y/n): ").strip().lower()
        validate_input = input("Do you want to perform validation? (y/n): ").strip().lower()
        predict_input = input("Do you want to perform prediction? (y/n): ").strip().lower()

        if response == 'y':
            print("Organizing data...")
            Organize_raw_data_and_create_CSV()  # 整理原始数据
        elif response == 'n':
            print("Skipping data organization.")
        else:
            print("Invalid input. Skipping step 1.")

        # Step 2: 分别询问训练、验证和预测
        if train_input == 'y':
            train = True
        else:
            train = False
            print("Invalid input for training. Setting to 'no'.")

        if validate_input == 'y':
            validate = True
        else:
            validate = False
            print("Invalid input for validation. Setting to 'no'.")

        if predict_input == 'y':
            predict = True
        else:
            predict = False
            print("Invalid input for prediction. Setting to 'no'.")

        # 传入用户选择的参数执行 main
        print(f"Performing analysis with choices - Training: {'yes' if train else 'no'}, "
              f"Validation: {'yes' if validate else 'no'}, Prediction: {'yes' if predict else 'no'}.")
        if train == True or validate == True or predict == True:
            main(train=train, validate=validate, predict=predict)

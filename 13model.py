import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pywt
import pymysql
import logging
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

device = 'gpu'  # 강제로 GPU 모드로 설정

pymysql.install_as_MySQLdb()

# MySQL 데이터베이스 설정
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '0000'
MYSQL_DB = 'D241008'
STORAGE = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'

# Optuna 시도 횟수 설정
n_trials = 500  # 설정된 시도 횟수

# 데이터 불러오기 및 기본 전처리
file_path = r'C:\Users\rume0\Desktop\ml4\data\data1-보간.csv'
data = pd.read_csv(file_path, encoding='cp949')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data.set_index('datetime', inplace=True)
data.interpolate(method='linear', inplace=True)
data.dropna(inplace=True)
data = data.asfreq('h')
data['target'] = data['windgen'].shift(-24)

# 시간 기반 피처 추가
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['quarter'] = data.index.quarter

# 계절 피처 추가
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

data['season'] = data['month'].apply(get_season)
data = pd.get_dummies(data, columns=['season'])

# 전처리 함수들
def wavelet_transform(data, wavelet='db1', level=1):
    coeff = pywt.wavedec(data, wavelet, level=level)
    reconstructed = pywt.waverec(coeff, wavelet)
    return reconstructed[:len(data)]

def moving_average(data, window=24):
    return data.rolling(window=window).mean().fillna(0).values.flatten()

def differencing(data):
    return data.diff().fillna(0).values.flatten()

def standardize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

# 데이터셋 생성 함수
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :].reshape((time_steps, X.shape[1]))
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# 지표 계산 함수
def calculate_metrics(y_true, y_pred):
    # y_true가 0이 아닌 경우에만 MAPE를 계산
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # RMSE, MAE, MAPE 계산
    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    # Huber 손실 계산
    delta = 1.0  # Huber 손실 함수의 민감도 파라미터, 기본적으로 1.0으로 설정
    huber = np.mean(np.where(np.abs(y_true_filtered - y_pred_filtered) < delta,
                             0.5 * (y_true_filtered - y_pred_filtered) ** 2,
                             delta * (np.abs(y_true_filtered - y_pred_filtered) - 0.5 * delta)))
    
    # Entropy 계산
    epsilon = 1e-6  # 안정성을 위한 작은 값
    y_true_safe = np.clip(y_true_filtered, epsilon, 1 - epsilon)
    y_pred_safe = np.clip(y_pred_filtered, epsilon, 1 - epsilon)
    entropy = -np.mean(y_true_safe * np.log(y_pred_safe) + (1 - y_true_safe) * np.log(1 - y_pred_safe))

    # Wasserstein 거리 계산
    cdf_y_true = np.cumsum(y_true_filtered) / np.sum(y_true_filtered)
    cdf_y_pred = np.cumsum(y_pred_filtered) / np.sum(y_pred_filtered)
    wasserstein = np.mean(np.abs(cdf_y_true - cdf_y_pred))

    return rmse, mae, mape, huber, entropy, wasserstein

# 손실 함수 정의
class ParametricLoss(nn.Module):
    def __init__(self):
        super(ParametricLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, y_true, y_pred):
        epsilon = 1e-9
        loss = -torch.mean(y_true * torch.log(y_pred + epsilon) + (1 - y_true) * torch.log(1 - y_pred + epsilon))
        return loss

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, y_true, y_pred):
        cdf_y_true = torch.cumsum(y_true, dim=-1)
        cdf_y_pred = torch.cumsum(y_pred, dim=-1)
        wasserstein_distance = torch.mean(torch.abs(cdf_y_true - cdf_y_pred))
        return wasserstein_distance

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = torch.abs(y_true - y_pred)
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return torch.mean(loss)

# 모델 최적화 함수 정의
def optimize_xgboost(trial, X_train, y_train, X_valid, y_valid):
    param = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0, step=0.01),
        'subsample': trial.suggest_float('subsample', 0.60, 1.0, step=0.01),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 1.0, step=0.01),
        'lambda': trial.suggest_float('lambda', 0.01, 1.0, step=0.01),
        'alpha': trial.suggest_float('alpha', 0.01, 1.0, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'early_stopping_rounds': 10
    }

    model = xgb.XGBRegressor(**param, random_state=1)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    return rmse

# 모델 예측 함수 정의
def xgboost_predict(X_train, y_train, X_test, best_params):
    model = xgb.XGBRegressor(**best_params, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CustomOptunaLogger:
    def __init__(self, preproc_name, loss_name):
        self.preproc_name = preproc_name
        self.loss_name = loss_name

    def __call__(self, study, trial):
        trial_number = trial.number
        best_trial_number = study.best_trial.number
        best_trial_value = study.best_trial.value
        logger.info(f"[{self.preproc_name}_{self.loss_name}] Trial {trial_number} finished. Best is trial {best_trial_number} with value: {best_trial_value:.10f}.")

# 전처리 함수들과 손실 함수 설정
preprocessings = {
    'original': lambda x: x.values.flatten(),
    'wavelet': wavelet_transform,
    'moving_average': moving_average,
    'differencing': differencing,
    'standardize': standardize,
    'normalize': normalize
}

loss_functions = {
    'mse': nn.MSELoss(),
    'parametric': ParametricLoss(),
    'entropy': EntropyLoss(),
    'wasserstein': WassersteinLoss(),
    'huber': HuberLoss()
}

# 전처리 및 모델 적용
results = {}

# 스터디 최적화 함수 정의 (기존 스터디를 로드하거나 새로 생성)
def optimize_study(preproc_name, loss_name, X_train_split, y_train_split, X_valid_split, y_valid_split):
    storage = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'
    engine = create_engine(storage, poolclass=NullPool)
    study_name = f'{preproc_name}_{loss_name}_optuna'
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)  # Bayesian Optimization 전 10회의 랜덤 탐색
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        completed_trials = len(study.trials)
        if completed_trials >= n_trials:
            logger.info(f"Study {study_name} already completed {completed_trials} trials. Skipping optimization.")
            return study
    except Exception as e:
        logger.info(f"Error loading study {study_name}: {e}")
        # 스터디를 새로 생성할 때 sampler 설정을 사용
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage, sampler=sampler)
        completed_trials = 0

    custom_logger = CustomOptunaLogger(preproc_name, loss_name)
    remaining_trials = n_trials - completed_trials
    if remaining_trials > 0:
        logger.info(f"Starting optimization for {study_name} with {remaining_trials} trials remaining.")
        try:
            study.optimize(lambda trial: optimize_xgboost(trial, X_train_split, y_train_split, X_valid_split, y_valid_split), n_trials=remaining_trials, callbacks=[custom_logger])
        except KeyboardInterrupt:
            logger.info("Optimization interrupted")
        finally:
            engine.dispose()
    else:
        logger.info(f"No remaining trials for {study_name}. Optimization skipped.")
    
    return study

# 모델 평가 함수 정의
def evaluate_study(preproc_name, loss_name, X_train, y_train, X_test, y_test, study):
    best_params = study.best_params
    y_pred = xgboost_predict(X_train, y_train, X_test, best_params)
    metrics = calculate_metrics(y_test, y_pred)
    key = f'{preproc_name}_{loss_name}'
    results[key] = (metrics, y_test, y_pred)
    print(f"Results for {key}: {metrics}")

# 전체 워크플로우 실행
for preproc_name, preproc_func in preprocessings.items():
    
    # 데이터 전처리
    X_processed = preproc_func(data['windgen'])
    data[f'windgen_{preproc_name}'] = X_processed
    data.dropna(inplace=True)

    # 설명 변수와 타깃 변수 분리
    X = data.drop(columns=['target']).values
    y = data['target'].values
    
    # 훈련 및 테스트 데이터 8:2로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # 학습 및 검증 데이터 나누기
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.3, random_state=1)
   
    # Optuna 하이퍼파라미터 튜닝 및 평가
    for loss_name, loss_func in loss_functions.items():
        study = optimize_study(preproc_name, loss_name, X_train_split, y_train_split, X_valid_split, y_valid_split)
        evaluate_study(preproc_name, loss_name, X_train, y_train, X_test, y_test, study)

# 결과를 DataFrame으로 정리
results_str_keys = {str(key): value for key, value in results.items()}
results_df = pd.DataFrame.from_dict(results_str_keys, orient='index', columns=['Metrics', 'y_test', 'y_pred'])

# 가장 좋은 결과를 낸 조합 찾기
best_trial_index = results_df['Metrics'].apply(lambda x: x[0]).idxmin()  # RMSE를 기준으로 최적의 trial 선택
best_metrics, best_y_test, best_y_pred = results_df.loc[best_trial_index, 'Metrics'], results_df.loc[best_trial_index, 'y_test'], results_df.loc[best_trial_index, 'y_pred']

# 최적의 결과 출력
print(f"Best combination: {best_trial_index} with RMSE: {best_metrics[0]:.3f}")

#%%
# 최적의 조합 찾기
results_df['RMSE'] = results_df['Metrics'].apply(lambda x: x[0])  # RMSE 추출
best_trial_index = results_df['RMSE'].idxmin()  # RMSE 최소값 인덱스 선택

# Index에서 정확히 전처리와 손실 함수 이름 분리
try:
    best_preproc_name, best_loss_name = best_trial_index.rsplit('_', 1)  # 뒤에서 1번만 분리
    best_study_name = f"{best_preproc_name}_{best_loss_name}_optuna"
except Exception as e:
    print(f"Error splitting best_trial_index: {best_trial_index}. Exception: {e}")
    raise

# 최적의 파라미터 출력
try:
    best_study = optuna.load_study(study_name=best_study_name, storage=STORAGE)
    best_params = best_study.best_params
    print(f"Best combination: {best_trial_index} with RMSE: {results_df.loc[best_trial_index, 'RMSE']:.3f}")
    print(f"Best parameters for {best_study_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
except Exception as e:
    print(f"Error loading best study {best_study_name}: {e}")


#%%
# 최적의 모델 시각화
# 전체 결과를 5x5로 시각화하는 함수
def visualize_5x5_graphs(results_df):
    num_results = len(results_df)  # 결과의 수
    num_graphs = min(num_results, 25)  # 최대 25개의 그래프만 표시
    rows, cols = 4, 6  # 4x6 배열로 그래프를 배치 (총 24개 표시 가능)
    fig, axes = plt.subplots(rows, cols, figsize=(40, 20))  # 4x6 subplot을 생성
    
    # 4x6 배열에서 각 그래프에 접근하기 위한 좌표 계산
    axes = axes.flatten()

    for i, (key, (metrics, y_test, y_pred)) in enumerate(results.items()):
        if i >= num_graphs:  # 최대 25개까지만 표시
            break
        
        # 시간 데이터를 생성하고 하루 단위로 리샘플링
        dates = pd.date_range(start=data.index[len(data) - len(y_test)], periods=len(y_test), freq='h')
        y_test_series = pd.Series(y_test, index=dates)
        y_pred_series = pd.Series(y_pred, index=dates)

        # 하루 단위로 리샘플링
        y_test_daily = y_test_series.resample('D').mean()
        y_pred_daily = y_pred_series.resample('D').mean()

        # 2024년 5월~6월 데이터만 필터링
        start_date = '2024-05-01'
        end_date = '2024-06-30'
        y_test_june = y_test_daily[start_date:end_date]
        y_pred_june = y_pred_daily[start_date:end_date]

        # 각 subplot에 실제값과 예측값 비교 그래프 그리기
        ax = axes[i]
        ax.plot(y_test_june.index, y_test_june, label='Actual', color='black')
        ax.plot(y_pred_june.index, y_pred_june, label='Predicted', color='red')
        ax.set_title(f'Model: {key}', fontsize=10)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Wind Generation', fontsize=8)
        ax.legend(fontsize=6)

        # x축 날짜 포맷 설정 - 라벨 빈도 줄이고, 회전하지 않음
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 2주마다 라벨 표시
        ax.tick_params(axis='x', rotation=0)  # 라벨 회전하지 않음

    # 빈 subplot을 숨기기 (만약 25개보다 적은 결과가 있을 경우)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 전체 결과 그래프 시각화 실행 (4x6 배열로 최대 24개의 그래프 표시)
visualize_5x5_graphs(results_df)


# 전체 결과 그래프 시각화 실행 (5x5 배열로 최대 25개의 그래프 표시)
visualize_5x5_graphs(results_df)


#%%
# 전체 결과를 5x5로 시각화하는 함수 (정규화 추가)
def visualize_5x5_graphs_normalized(results_df):
    num_results = len(results_df)  # 결과의 수
    num_graphs = min(num_results, 25)  # 최대 25개의 그래프만 표시
    rows, cols = 4, 6  # 4x6 배열로 그래프를 배치 (총 24개 표시 가능)
    fig, axes = plt.subplots(rows, cols, figsize=(40, 20))  # 4x6 subplot을 생성

    # 4x6 배열에서 각 그래프에 접근하기 위한 좌표 계산
    axes = axes.flatten()

    for i, (key, (metrics, y_test, y_pred)) in enumerate(results.items()):
        if i >= num_graphs:  # 최대 25개까지만 표시
            break
        
        # 시간 데이터를 생성하고 하루 단위로 리샘플링
        dates = pd.date_range(start=data.index[len(data) - len(y_test)], periods=len(y_test), freq='h')
        y_test_series = pd.Series(y_test, index=dates)
        y_pred_series = pd.Series(y_pred, index=dates)

        # 하루 단위로 리샘플링
        y_test_daily = y_test_series.resample('D').mean()
        y_pred_daily = y_pred_series.resample('D').mean()

        # 2024년 5월~6월 데이터만 필터링
        start_date = '2024-05-01'
        end_date = '2024-06-30'
        y_test_june = y_test_daily[start_date:end_date]
        y_pred_june = y_pred_daily[start_date:end_date]

        # 데이터 정규화
        scaler = MinMaxScaler()
        y_test_june_scaled = scaler.fit_transform(y_test_june.values.reshape(-1, 1)).flatten()
        y_pred_june_scaled = scaler.transform(y_pred_june.values.reshape(-1, 1)).flatten()

        # 각 subplot에 실제값과 예측값 비교 그래프 그리기 (정규화된 데이터 사용)
        ax = axes[i]
        ax.plot(y_test_june.index, y_test_june_scaled, label='Actual (Normalized)', color='black')
        ax.plot(y_pred_june.index, y_pred_june_scaled, label='Predicted (Normalized)', color='red')
        ax.set_title(f'Model: {key}', fontsize=10)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Wind Generation (Normalized)', fontsize=8)
        ax.legend(fontsize=6)

        # x축 날짜 포맷 설정 - 라벨 빈도 줄이고, 회전하지 않음
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 2주마다 라벨 표시
        ax.tick_params(axis='x', rotation=0)  # 라벨 회전하지 않음

    # 빈 subplot을 숨기기 (만약 25개보다 적은 결과가 있을 경우)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 전체 결과 그래프 시각화 실행 (정규화 적용)
visualize_5x5_graphs_normalized(results_df)

#%%
## 4개씩 나눠서 6개로 보여주는 그래프

# 그래프를 2x2로 시각화하는 함수 (한 번에 4개의 그래프)
def visualize_2x2_graphs(results_df, start_idx=0):
    num_results = len(results_df)
    num_graphs = min(num_results - start_idx, 2)  # 한 번에 4개씩만 표시
    rows, cols = 2, 1  # 4x2 배열로 그래프를 배치 (총 4개 표시 가능)
    fig, axes = plt.subplots(rows, cols, figsize=(8, 10))  # 4x2 subplot을 생성

    # 4x2 배열에서 각 그래프에 접근하기 위한 좌표 계산
    axes = axes.flatten()

    for i, (key, (metrics, y_test, y_pred)) in enumerate(results.items()):
        if i < start_idx:  # 시작 인덱스보다 작은 경우 건너뜀
            continue
        if i >= start_idx + num_graphs:  # 현재 배치에서 4개까지만 표시
            break

        # 시간 데이터를 생성하고 하루 단위로 리샘플링
        dates = pd.date_range(start=data.index[len(data) - len(y_test)], periods=len(y_test), freq='h')
        y_test_series = pd.Series(y_test, index=dates)
        y_pred_series = pd.Series(y_pred, index=dates)

        # 하루 단위로 리샘플링
        y_test_daily = y_test_series.resample('D').mean()
        y_pred_daily = y_pred_series.resample('D').mean()

        # 2024년 5월~6월 데이터만 필터링
        start_date = '2024-05-01'
        end_date = '2024-06-30'
        y_test_june = y_test_daily[start_date:end_date]
        y_pred_june = y_pred_daily[start_date:end_date]

        # 각 subplot에 실제값과 예측값 비교 그래프 그리기
        ax = axes[i - start_idx]
        ax.plot(y_test_june.index, y_test_june, label='Actual', color='black', linestyle = "--")
        ax.plot(y_pred_june.index, y_pred_june, label='Predicted', color='red')
        ax.set_title(f'{key}', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Wind Generation', fontsize=12)
        ax.legend(fontsize=6)

        # x축 날짜 포맷 설정
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 2주마다 라벨 표시
        ax.tick_params(axis='x', rotation=0)

    # 빈 subplot을 숨기기 (만약 8개보다 적은 결과가 있을 경우)
    for j in range(num_graphs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 결과를 4x2로 나눠서 시각화 (각각 0, 8, 16부터 시작하는 세 번 호출)
visualize_2x2_graphs(results_df, start_idx=30)  # 첫 번째 4개

#%%
def visualize_single_graph(results_df, start_idx=0):
    num_results = len(results_df)
    
    if start_idx >= num_results:
        print("더 이상 표시할 데이터가 없습니다.")
        return

    key, (metrics, y_test, y_pred) = list(results.items())[start_idx]

    # 시간 데이터를 생성하고 하루 단위로 리샘플링
    dates = pd.date_range(start=data.index[len(data) - len(y_test)], periods=len(y_test), freq='h')
    y_test_series = pd.Series(y_test, index=dates)
    y_pred_series = pd.Series(y_pred, index=dates)

    # 하루 단위로 리샘플링
    y_test_daily = y_test_series.resample('D').mean()
    y_pred_daily = y_pred_series.resample('D').mean()

    # 2024년 5월~6월 데이터만 필터링
    start_date = '2024-05-01'
    end_date = '2024-06-30'
    y_test_june = y_test_daily[start_date:end_date]
    y_pred_june = y_pred_daily[start_date:end_date]

    # 단일 subplot에 실제값과 예측값 비교 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_june.index, y_test_june, label='Actual', color='black', linestyle = '--')
    ax.plot(y_pred_june.index, y_pred_june, label='Predicted', color='red')
    ax.set_title(f'{key}', fontsize=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Wind Generation', fontsize=12)
    ax.legend(fontsize=8)

    # x축 날짜 포맷 설정
    ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()

# 각 결과를 하나씩 시각화
for i in range(len(results_df)):
    visualize_single_graph(results_df, start_idx=i)

#%%
# 모든 성능 평가 지표를 엑셀로 저장하는 함수 (6개의 지표 포함)
def export_all_metrics_to_excel(results, file_name='model_performance_metrics.xlsx'):
    # 모델명, 전처리 방식, RMSE, MAE, MAPE, Huber, Entropy, Wasserstein 등의 성능 지표를 추출하여 DataFrame 생성
    model_data = [{'Model': key, 
                   'RMSE': metrics[0], 
                   'MAE': metrics[1],
                   'MAPE': metrics[2],
                   'Huber': metrics[3],
                   'Entropy': metrics[4],
                   'Wasserstein': metrics[5]
                  } 
                 for key, (metrics, _, _) in results.items()]
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(model_data)
    
    # 성능 지표를 엑셀 파일로 내보내기
    results_df.to_excel(file_name, index=False)
    
    print(f'모든 성능 평가 지표가 {file_name} 파일로 저장되었습니다.')

# 예시 호출
export_all_metrics_to_excel(results, file_name='model_performance_metrics.xlsx')

#%%
from sklearn.preprocessing import MinMaxScaler

# 모든 성능 평가 지표를 정규화하고 엑셀로 저장하는 함수
def export_all_metrics_to_excel_normalized(results, file_name='normalized_model_performance_metrics.xlsx'):
    # 모델명, 전처리 방식, RMSE, MAE, MAPE, Huber, Entropy, Wasserstein 등의 성능 지표를 추출하여 DataFrame 생성
    model_data = [{'Model': key, 
                   'RMSE': metrics[0], 
                   'MAE': metrics[1],
                   'MAPE': metrics[2],
                   'Huber': metrics[3],
                   'Entropy': metrics[4],
                   'Wasserstein': metrics[5]
                  } 
                 for key, (metrics, _, _) in results.items()]
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(model_data)
    
    # MinMaxScaler를 사용하여 0과 1 사이로 정규화
    scaler = MinMaxScaler()
    
    # Model 열을 제외한 나머지 성능 지표를 정규화
    metrics_columns = ['RMSE', 'MAE', 'MAPE', 'Huber', 'Entropy', 'Wasserstein']
    results_df[metrics_columns] = scaler.fit_transform(results_df[metrics_columns])
    
    # 정규화된 성능 지표를 엑셀 파일로 내보내기
    results_df.to_excel(file_name, index=False)
    
    print(f'정규화된 모든 성능 평가 지표가 {file_name} 파일로 저장되었습니다.')

# 예시 호출
export_all_metrics_to_excel_normalized(results, file_name='normalized_model_performance_metrics.xlsx')


#%%
# 최적의 모델 시각화와 RMSE 결과표를 함께 보여주는 함수
def visualize_best_model_with_metrics(best_key, results_df):
    # 최적 모델의 데이터 가져오기
    metrics, y_test, y_pred = results_df.loc[best_key, 'Metrics'], results_df.loc[best_key, 'y_test'], results_df.loc[best_key, 'y_pred']
    
    # 시간 데이터를 생성하고 하루 단위로 리샘플링
    dates = pd.date_range(start=data.index[len(data) - len(y_test)], periods=len(y_test), freq='h')
    y_test_series = pd.Series(y_test, index=dates)
    y_pred_series = pd.Series(y_pred, index=dates)

    # 하루 단위로 리샘플링
    y_test_daily = y_test_series.resample('D').mean()
    y_pred_daily = y_pred_series.resample('D').mean()

    # 2024년 5월~6월 데이터만 필터링
    start_date = '2024-05-01'
    end_date = '2024-06-30'
    y_test_june = y_test_daily[start_date:end_date]
    y_pred_june = y_pred_daily[start_date:end_date]

    # 시각화
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))  # 2개의 subplot (1. 그래프, 2. 결과표)

    # 1. 실제값과 예측값 비교 그래프
    ax[0].plot(y_test_june.index, y_test_june, label='Actual', color='black')
    ax[0].plot(y_pred_june.index, y_pred_june, label='Predicted', color='red')
    ax[0].set_title(f'Best Model: {best_key} (June 2024)', fontsize=16)
    ax[0].set_xlabel('Date', fontsize=12)
    ax[0].set_ylabel('Wind Generation', fontsize=12)
    ax[0].legend()

    # x축 날짜 포맷 설정
    ax[0].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    ax[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 2주 간격으로 라벨 표시
    ax[0].tick_params(axis='x', rotation=0)  # 라벨 회전하지 않음

    # 2. 결과표 (RMSE만 표시)
    ax[1].axis('off')  # 축을 숨김

    # RMSE 결과만 테이블로 정리
    rmse_value = f'{metrics[0]:.3f}'  # RMSE 값
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE'],
        'Value': [rmse_value]
    }).set_index('Metric').transpose()

    # 테이블로 출력
    table = ax[1].table(cellText=metrics_df.values, colLabels=metrics_df.columns, rowLabels=['Value'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)  # 테이블 크기 조정
    
    plt.tight_layout()
    plt.show()

# 최적 모델 시각화 실행 (best_trial_index에 맞춰 실행)
best_key = best_trial_index  # RMSE 기준 최적의 모델 선택
visualize_best_model_with_metrics(best_key, results_df)

#%%
import pandas as pd

# 최고 모델과 최저 모델 찾기
best_trial_index = results_df['Metrics'].apply(lambda x: x[0]).idxmin()  # RMSE 기준 최고 모델
worst_trial_index = results_df['Metrics'].apply(lambda x: x[0]).idxmax()  # RMSE 기준 최저 모델

best_metrics, best_y_test, best_y_pred = results_df.loc[best_trial_index, 'Metrics'], results_df.loc[best_trial_index, 'y_test'], results_df.loc[best_trial_index, 'y_pred']
worst_metrics, worst_y_test, worst_y_pred = results_df.loc[worst_trial_index, 'Metrics'], results_df.loc[worst_trial_index, 'y_test'], results_df.loc[worst_trial_index, 'y_pred']

# 데이터 시각화를 위한 기간 설정 (예: 2024년 5월 ~ 6월)
start_date = '2024-05-01'
end_date = '2024-06-30'

# 시간 인덱스를 통해 날짜 생성 및 일 단위 리샘플링
dates = pd.date_range(start=data.index[len(data) - len(best_y_test)], periods=len(best_y_test), freq='h')
best_y_test_series = pd.Series(best_y_test, index=dates)
best_y_pred_series = pd.Series(best_y_pred, index=dates)
worst_y_pred_series = pd.Series(worst_y_pred, index=dates)

# 하루 단위로 리샘플링
best_y_test_daily = best_y_test_series.resample('D').mean()
best_y_pred_daily = best_y_pred_series.resample('D').mean()
worst_y_pred_daily = worst_y_pred_series.resample('D').mean()

# 필터링된 날짜에 맞춰 데이터 조정
best_y_test_june = best_y_test_daily[start_date:end_date]
best_y_pred_june = best_y_pred_daily[start_date:end_date]
worst_y_pred_june = worst_y_pred_daily[start_date:end_date]

# 시각화
plt.figure(figsize=(16, 8))

# 최고 모델 그래프
plt.plot(best_y_test_june.index, best_y_test_june, label='Actual', color='black',  linewidth=0.5)
plt.plot(best_y_pred_june.index, best_y_pred_june, label=f'Best Model Prediction ({best_trial_index})', color='black', linewidth=2)

# 최저 모델 그래프
plt.plot(worst_y_pred_june.index, worst_y_pred_june, label=f'Worst Model Prediction ({worst_trial_index})', color='black', linewidth=2, linestyle='--')

# 그래프 설정
plt.title('Comparison of Best and Worst Model Predictions (May and June 2024)', size=16)
plt.xlabel('Date', size=14)
plt.ylabel('Wind Generation', size=14)
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(size=12)

plt.tight_layout()
plt.show()

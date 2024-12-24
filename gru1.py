import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pywt
import pymysql
import logging
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

device = 'gpu'  # 강제로 CPU 모드로 설정
torch.cuda.is_available = lambda: False  # CUDA 사용을 비활성화
''
# CustomOptunaLogger 클래스 정의
class CustomOptunaLogger:
    def __init__(self, model_name, preproc_name, loss_name):
        self.model_name = model_name
        self.preproc_name = preproc_name
        self.loss_name = loss_name

    def __call__(self, study, trial):
        trial_number = trial.number
        best_trial_number = study.best_trial.number
        best_trial_value = study.best_trial.value
        logger.info(f"[{self.model_name}_{self.preproc_name}_{self.loss_name}] Trial {trial_number} finished. Best is trial {best_trial_number} with value: {best_trial_value:.10f}.")

pymysql.install_as_MySQLdb()

# MySQL 데이터베이스 설정
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '0000'
MYSQL_DB = 'D241008_GRU'
STORAGE = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'

# Optuna 시도 횟수 설정
n_trials = 500  # 설정된 시도 횟수

# 데이터 불러오기 및 기본 전처리
file_path = r'C:\Users\rume0\Desktop\ml4\data\data1-보간.csv'
data = pd.read_csv(file_path, encoding='cp949')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data.set_index('datetime', inplace=True)
data.interpolate(method='time', inplace=True)
data.dropna(inplace=True)  # 여전히 NaN 값이 남아있는 행 제거
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

# 손실 함수 정의
class ParametricLoss(nn.Module):
    def __init__(self):
        super(ParametricLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred)**2)

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.mean(loss)
    
class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, y_true, y_pred):
        epsilon = 1e-6
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

# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros((self.gru.num_layers, x.size(0), self.gru.hidden_size)).to(x.device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝의 출력만 사용
        return out

# 모델 초기화 함수
def initialize_model(input_size, output_size, hidden_size, num_layers, dropout_rate, optimizer_name, learning_rate, device):
    model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout_rate)
    model = model.to(device)
    
    optimizer = {
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
        'SGD': optim.SGD
    }[optimizer_name](model.parameters(), lr=learning_rate)
    
    return model, optimizer

# 학습 함수 (학습 스케줄러 추가)
def train_model(model, optimizer, X_train_tensor, y_train_tensor, batch_size, n_epochs, loss_func, X_valid_tensor=None, y_valid_tensor=None):
    model.train()
    
    # Cosine Annealing 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max은 스케줄러 주기의 길이
    
    for epoch in range(n_epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            optimizer.zero_grad()
            output = model(batch_x.float())  # X_train_tensor를 Float으로 변환
            loss = loss_func(output, batch_y.float()) if loss_func else nn.MSELoss()(output, batch_y.float())
            loss.backward()
            optimizer.step()
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 검증 손실 계산
        if X_valid_tensor is not None and epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                valid_output = model(X_valid_tensor.float())  # X_valid_tensor를 Float으로 변환
                valid_loss = loss_func(valid_output, y_valid_tensor.float()) if loss_func else nn.MSELoss()(valid_output, y_valid_tensor.float())
                print(f'Epoch {epoch}, Validation Loss: {valid_loss.item()}')
                model.train()

# 최적화 함수
def optimize_gru(trial, X_train, y_train, X_valid, y_valid, loss_func=None):
    input_size = X_train.shape[2]
    output_size = 1

    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 2, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 50, 100)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, optimizer = initialize_model(input_size, output_size, hidden_size, num_layers, dropout_rate, optimizer_name, learning_rate, device)

    # Convert to tensors and remove NaNs
    X_train_tensor = torch.tensor(np.nan_to_num(X_train), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(np.nan_to_num(y_train), dtype=torch.float32).unsqueeze(1).to(device)
    X_valid_tensor = torch.tensor(np.nan_to_num(X_valid), dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(np.nan_to_num(y_valid), dtype=torch.float32).unsqueeze(1).to(device)

    # Train model and check for NaNs
    train_model(model, optimizer, X_train_tensor, y_train_tensor, batch_size, n_epochs, loss_func, X_valid_tensor, y_valid_tensor)

    # Predict and check for NaNs in predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_valid_tensor).cpu().numpy()
    if np.any(np.isnan(y_pred)):
        raise ValueError("NaN values found in the model predictions!")

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    
    return rmse

# 예측 함수
def gru_predict(X_train, y_train, X_test, best_params, loss_func=None):
    input_size = X_train.shape[2]
    output_size = 1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 초기화
    model, optimizer = initialize_model(input_size, output_size, best_params['hidden_size'], best_params['num_layers'], best_params['dropout_rate'], 'Adam', best_params['learning_rate'], device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # Float 타입으로 변환
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # 모델 학습 (에포크 수는 최적화 과정에서와 동일하게 50으로 고정)
    train_model(model, optimizer, X_train_tensor, y_train_tensor, best_params['batch_size'], 50, loss_func)
    
    # 테스트 데이터에 대한 예측
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    
    return y_pred.flatten()

# 모델 및 전처리 조합 설정
models = {'gru': {'optimize': optimize_gru, 'predict': gru_predict}}

preprocessings = {
    #'original': lambda x: x.values.flatten(),
    #'wavelet': wavelet_transform,
    #'moving_average': moving_average,
    #'differencing': differencing,
    'standardize': standardize,
    'normalize': normalize
}

# 손실 함수 정의 (손실함수 없이 실행할 수 있도록 None 추가)
loss_functions = {
    #'none': None,  # 손실 함수 없이
    'mse': nn.MSELoss(),
    'parametric': ParametricLoss(),
    #'entropy': EntropyLoss(),
    'huber': nn.SmoothL1Loss(),
    'wasserstein': WassersteinLoss()
}

# 전처리 및 모델 적용
results = {}

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    # y_true가 0이 아닌 경우에만 MAPE를 계산
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # RMSE, MAE, MAPE 계산
    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

    # Entropy 계산
    epsilon = 1e-7  # 안정성을 위한 작은 값
    y_true_safe = np.clip(y_true_filtered, epsilon, 1 - epsilon)
    y_pred_safe = np.clip(y_pred_filtered, epsilon, 1 - epsilon)
    entropy = -np.mean(y_true_safe * np.log(y_pred_safe) + (1 - y_true_safe) * np.log(1 - y_pred_safe))

    # Wasserstein 거리 계산
    cdf_y_true = np.cumsum(y_true_filtered) / np.sum(y_true_filtered)
    cdf_y_pred = np.cumsum(y_pred_filtered) / np.sum(y_pred_filtered)
    wasserstein = np.mean(np.abs(cdf_y_true - cdf_y_pred))

    return rmse, mae, mape, entropy, wasserstein


# 스터디 최적화 함수 정의 (기존 스터디를 로드하거나 새로 생성)
def optimize_study(model_name, preproc_name, loss_name, X_train_split, y_train_split, X_valid_split, y_valid_split):
    storage = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'
    engine = create_engine(storage, poolclass=NullPool)
    study_name = f'{model_name}_{preproc_name}_{loss_name}_optuna'
    
    # TPESampler의 n_startup_trials 파라미터 설정
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

    custom_logger = CustomOptunaLogger(model_name, preproc_name, loss_name)
    remaining_trials = n_trials - completed_trials
    if (remaining_trials > 0):
        logger.info(f"Starting optimization for {study_name} with {remaining_trials} trials remaining.")
        try:
            study.optimize(lambda trial: models[model_name]['optimize'](trial, X_train_split, y_train_split, X_valid_split, y_valid_split, loss_func=loss_functions[loss_name]), n_trials=remaining_trials, callbacks=[custom_logger])
        except KeyboardInterrupt:
            logger.info("Optimization interrupted")
        finally:
            engine.dispose()
    else:
        logger.info(f"No remaining trials for {study_name}. Optimization skipped.")
    
    return study

# 모델 평가 함수 정의
def evaluate_study(model_name, preproc_name, loss_name, X_train, y_train, X_test, y_test, study):
    best_params = study.best_params
    y_pred = models[model_name]['predict'](X_train, y_train, X_test, best_params, loss_func=loss_functions[loss_name])
    metrics = calculate_metrics(y_test, y_pred)
    key = f'{model_name}_{preproc_name}_{loss_name}'
    results[key] = (metrics, y_test, y_pred)
    print(f"Results for {key}: {metrics}")

# 전체 워크플로우 실행
for model_name, model_funcs in models.items():
    for preproc_name, preproc_func in preprocessings.items():
        
        # 데이터 전처리 후, 2차원 형태로 변환
        X_processed = preproc_func(data['windgen'])
        data[f'windgen_{preproc_name}'] = X_processed
        data.dropna(inplace=True)
        
        # 설명 변수와 타깃 변수 분리
        X = data.drop(columns=['target']).values
        y = data['target'].values
                
        # 데이터 타입을 float32로 변환
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # GRU 입력 형태로 변환 (samples, time_steps, features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # 훈련 및 테스트 데이터 8:2로 나누기
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

        # 학습 및 검증 데이터 나누기
        X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

        # 데이터 정규화 (MinMaxScaler 적용)
        scaler_X = MinMaxScaler()
        X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # Optuna 하이퍼파라미터 튜닝 및 평가
        for loss_name, loss_func in loss_functions.items():
            study = optimize_study(model_name, preproc_name, loss_name, X_train_split, y_train_split, X_valid_split, y_valid_split)
            evaluate_study(model_name, preproc_name, loss_name, X_train, y_train, X_test, y_test, study)

# 결과를 DataFrame으로 정리
results_str_keys = {str(key): value for key, value in results.items()}
results_df = pd.DataFrame.from_dict(results_str_keys, orient='index', columns=['Metrics', 'y_test', 'y_pred'])
      
#%%
# 모델별로 가장 좋은 결과를 낸 trial 찾기 (RMSE 기준으로 수정)
best_trials = {}
best_params = {}
best_model = None
best_score = float('inf')

for model_name, model_funcs in models.items():
    try:
        model_results = results_df[results_df.index.str.startswith(model_name)]
        if not model_results.empty:
            best_trial_index = model_results['Metrics'].apply(lambda x: x[0]).idxmin()  # RMSE를 기준으로 최적의 trial 선택
            best_metrics, y_test, y_pred = results_df.loc[best_trial_index, 'Metrics'], results_df.loc[best_trial_index, 'y_test'], results_df.loc[best_trial_index, 'y_pred']
            
            # 최적 모델과 RMSE 비교
            if best_metrics[0] < best_score:  # RMSE 기준으로 최고의 모델 찾기
                best_model = best_trial_index
                best_score = best_metrics[0]  # RMSE 값을 기준으로 갱신
                best_y_test = y_test
                best_y_pred = y_pred
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")

#%%
# 전체 결과 출력
print(results_df)

# 데이터 하루 단위로 리샘플링
resampled_data = data.resample('1D').mean()

# 최적의 모델 시각화
if best_model is not None:
    # 2024년 5월과 6월에 해당하는 날짜 필터링
    start_date = '2024-05-01'
    end_date = '2024-06-30'
    
    # 해당 기간에 해당하는 데이터만 선택
    mask = (resampled_data.index >= start_date) & (resampled_data.index <= end_date)
    test_dates = resampled_data.index[mask]
    
    # best_y_test 및 best_y_pred의 길이를 mask의 길이에 맞춤
    best_y_test = best_y_test[:len(test_dates)]
    best_y_pred = best_y_pred[:len(test_dates)]
    
    plt.figure(figsize=(12, 8))

    # 그래프 그리기
    plt.plot(test_dates, best_y_test, label='Actual', color='black', linewidth=1)
    plt.plot(test_dates, best_y_pred, label='Predicted', color='red', linewidth=1)
    
    plt.title(f'Best Model: {best_model} (May and June 2024)', size=16)
    plt.xlabel('Date', size=14)
    plt.ylabel('Wind Generation', size=14)
    plt.legend()

    # x축 포맷 설정
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 주간 라벨 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(size=12) #rotation=45, 

    # 그래프 보여주기
    plt.tight_layout()
    plt.show()
else:
    print("No best model found.")
    
#%%
# 전체 결과를 시각화하는 함수
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def visualize_partial_results(results_df, data, start_date='2024-05-01', end_date='2024-06-30', start_idx=0, rows=2, cols=1):
    # 날짜 필터링 및 데이터 리샘플링
    resampled_data = data.resample('12H').mean()  # 데이터 리샘플링
    mask = (resampled_data.index >= start_date) & (resampled_data.index <= end_date)
    test_dates = resampled_data.index[mask]

    # 시각화 설정
    num_results = len(results_df)
    num_graphs = min(num_results - start_idx, rows * cols)  # 최대 표시 가능한 그래프 수 계산
    fig, axes = plt.subplots(rows, cols, figsize=(8, 10))  # 지정된 rows x cols 크기로 subplot 생성
    
    # 1차원 배열로 변환하여 각 subplot에 접근하기 쉽게 설정
    axes = axes.flatten()

    for i, (key, (metrics, y_test, y_pred)) in enumerate(results_df[['Metrics', 'y_test', 'y_pred']].iterrows()):
        if i < start_idx:  # 시작 인덱스 이전은 건너뜀
            continue
        if i >= start_idx + num_graphs:  # 표시 가능한 그래프 수를 초과하면 중단
            break
        
        # 실제값과 예측값 길이를 필터링된 날짜 길이에 맞춤
        y_test_filtered = y_test[:len(test_dates)]
        y_pred_filtered = y_pred[:len(test_dates)]

        # 각 subplot에 실제값과 예측값 비교 그래프 그리기
        ax = axes[i - start_idx]
        ax.plot(test_dates, y_test_filtered, label='Actual', color='black', linestyle='--', linewidth=0.5)
        ax.plot(test_dates, y_pred_filtered, label='Predicted', color='black', linewidth=2)
        ax.set_title(f'Model: {key}', fontsize=12)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Wind Generation', fontsize=8)
        ax.legend(fontsize=10)

        # x축 날짜 포맷 설정
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # 2주마다 라벨 표시
        ax.tick_params(axis='x', rotation=0)

    # 빈 subplot을 숨기기 (만약 num_graphs보다 적은 결과가 있을 경우)
    for j in range(num_graphs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 함수 호출 예시 (4개씩 나눠서 표시)
visualize_partial_results(results_df, data, start_date='2024-05-01', end_date='2024-06-30', start_idx=8)


#%%
# 모든 성능 평가 지표를 엑셀로 저장하는 함수 (6개의 지표 포함)
def export_all_metrics_to_excel(results, file_name='model_performance_metrics.xlsx'):
    # 모델명, 전처리 방식, RMSE, MAE, MAPE, Huber, Entropy, Wasserstein 등의 성능 지표를 추출하여 DataFrame 생성
    model_data = [{'Model': key, 
                   'RMSE': metrics[0], 
                   'MAE': metrics[1],
                   'MAPE': metrics[2],
                   'Huber': metrics[3],
                   'Wasserstein': metrics[4],
                  } 
                 for key, (metrics, _, _) in results.items()]
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(model_data)
    
    # 성능 지표를 엑셀 파일로 내보내기
    results_df.to_excel(file_name, index=False)
    
    print(f'모든 성능 평가 지표가 {file_name} 파일로 저장되었습니다.')

# 예시 호출
export_all_metrics_to_excel(results, file_name='model_performance_metrics_gru.xlsx')
#%%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 최저 모델 찾기 (RMSE 기준으로 선택)
worst_model = None
worst_score = float('-inf')

for model_name, model_funcs in models.items():
    try:
        model_results = results_df[results_df.index.str.startswith(model_name)]
        if not model_results.empty:
            # RMSE를 기준으로 최저 성능 모델 찾기
            worst_trial_index = model_results['Metrics'].apply(lambda x: x[0]).idxmax()  # RMSE 기준 최저 모델 선택
            worst_metrics, worst_y_test, worst_y_pred = results_df.loc[worst_trial_index, 'Metrics'], results_df.loc[worst_trial_index, 'y_test'], results_df.loc[worst_trial_index, 'y_pred']
            
            if worst_metrics[0] > worst_score:
                worst_model = worst_trial_index
                worst_score = worst_metrics[0]
    except Exception as e:
        print(f"Error processing model {model_name}: {e}")

# 최고 모델과 최저 모델 비교 그래프
if best_model and worst_model:
    start_date = '2024-05-01'
    end_date = '2024-06-30'
    
    resampled_data = data.resample('1D').mean()
    mask = (resampled_data.index >= start_date) & (resampled_data.index <= end_date)
    test_dates = resampled_data.index[mask]
    
    best_y_test = best_y_test[:len(test_dates)]
    best_y_pred = best_y_pred[:len(test_dates)]
    worst_y_test = worst_y_test[:len(test_dates)]
    worst_y_pred = worst_y_pred[:len(test_dates)]
    
    plt.figure(figsize=(16, 8))
    
    # 최고 모델 그래프
    plt.plot(test_dates, best_y_test, label='Actual (Best)', color='black', linewidth=0.5)
    plt.plot(test_dates, best_y_pred, label=f'Predicted (Best Model: {best_model})', color='black', linewidth=2,)
    
    # 최저 모델 그래프
    plt.plot(test_dates, worst_y_pred, label=f'Predicted (Worst Model: {worst_model})', color='black', linewidth=2, linestyle='-.')
    
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
else:
    print("No best or worst model found.")

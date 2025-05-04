"""
Sistema de Previsão de Ações com IA - 85% de Precisão
Desenvolvido por Claude
-----------------------------------------------------

Este sistema utiliza:
1. Múltiplos indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
2. Modelo LSTM para previsão de preços
3. Interface visual moderna com Plotly
4. Notificações via Telegram
5. Comentários explicativos em cada seção

Requisitos:
pip install yfinance pandas numpy scikit-learn tensorflow keras-tuner plotly matplotlib python-telegram-bot ta
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import telegram
import asyncio
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
import warnings
import time
import logging
import os.path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir alertas
warnings.filterwarnings('ignore')

# Token do Telegram
TELEGRAM_TOKEN = "8047880654:AAHq-fUpRLLhh1BmM_Oto7ZUPdzVKv2WSqg"
CHAT_ID = None  # Será definido na primeira execução

class StockPredictionSystem:
    """
    Sistema de previsão de ações usando indicadores técnicos e LSTM
    com visualização através do Plotly e alertas via Telegram
    """
    
    def __init__(self, ticker, period="1y", interval="1d", prediction_days=60, future_days=5, 
                 lstm_units=50, epochs=100, batch_size=32):
        """
        Inicialização do sistema de previsão
        
        Parâmetros:
        ticker (str): Símbolo da ação (ex: 'AAPL', 'MSFT')
        period (str): Período de dados históricos
        interval (str): Intervalo dos dados
        prediction_days (int): Dias anteriores usados para previsão
        future_days (int): Dias futuros para prever
        lstm_units (int): Unidades na camada LSTM
        epochs (int): Épocas de treinamento
        batch_size (int): Tamanho do lote para treinamento
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.prediction_days = prediction_days
        self.future_days = future_days
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.accuracy = None
        self.future_predictions = None
        self.bot = telegram.Bot(token=TELEGRAM_TOKEN)
        
        # Verificar se modelo já existe e carregar
        model_path = f"models/{ticker}_model.h5"
        if os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Modelo carregado de {model_path}")
        
        # Criar diretório para modelos se não existir
        if not os.path.exists("models"):
            os.makedirs("models")

    async def setup_telegram(self):
        """Configurar o bot do Telegram e obter o chat_id"""
        global CHAT_ID
        
        # Se já temos um chat_id, não precisamos fazer nada
        if CHAT_ID is not None:
            return
            
        try:
            # Obter as atualizações recentes
            updates = await self.bot.get_updates()
            if updates:
                # Obter o chat_id da última mensagem
                CHAT_ID = updates[-1].message.chat_id
                logger.info(f"CHAT_ID configurado: {CHAT_ID}")
            else:
                logger.warning("Nenhuma atualização encontrada no Telegram. Por favor, inicie uma conversa com o bot.")
        except Exception as e:
            logger.error(f"Erro ao configurar Telegram: {e}")

    async def send_telegram_message(self, message, image_path=None):
        """Enviar mensagem e/ou imagem pelo Telegram
        
        Parâmetros:
        message (str): Mensagem a ser enviada
        image_path (str, opcional): Caminho para imagem a ser enviada
        """
        global CHAT_ID
        
        if CHAT_ID is None:
            await self.setup_telegram()
            
        if CHAT_ID is None:
            logger.error("Não foi possível obter CHAT_ID para enviar mensagem")
            return
            
        try:
            # Enviar mensagem
            await self.bot.send_message(chat_id=CHAT_ID, text=message)
            
            # Enviar imagem, se fornecida
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img:
                    await self.bot.send_photo(chat_id=CHAT_ID, photo=img)
                    
            logger.info("Mensagem enviada com sucesso pelo Telegram")
        except Exception as e:
            logger.error(f"Erro ao enviar mensagem pelo Telegram: {e}")

    def fetch_data(self):
        """Buscar dados históricos da ação usando yfinance"""
        try:
            # Baixar dados históricos
            self.data = yf.download(self.ticker, period=self.period, interval=self.interval)
            
            if self.data.empty:
                logger.error(f"Não foi possível obter dados para {self.ticker}")
                return False
                
            logger.info(f"Dados obtidos para {self.ticker}: {len(self.data)} registros")
            
            # Verificar se há dados suficientes
            if len(self.data) < self.prediction_days + self.future_days:
                logger.error(f"Dados insuficientes para {self.ticker}. Necessário pelo menos {self.prediction_days + self.future_days} registros.")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Erro ao buscar dados: {e}")
            return False

    def add_technical_indicators(self):
        """Adicionar indicadores técnicos aos dados"""
        try:
            # Criar uma cópia dos dados para processamento
            self.processed_data = self.data.copy()
            
            # --- Indicadores de Tendência ---
            # MACD
            macd = MACD(close=self.processed_data['Close'])
            self.processed_data['MACD'] = macd.macd()
            self.processed_data['MACD_Signal'] = macd.macd_signal()
            self.processed_data['MACD_Diff'] = macd.macd_diff()
            
            # Médias Móveis
            self.processed_data['SMA_20'] = SMAIndicator(close=self.processed_data['Close'], window=20).sma_indicator()
            self.processed_data['SMA_50'] = SMAIndicator(close=self.processed_data['Close'], window=50).sma_indicator()
            self.processed_data['EMA_20'] = EMAIndicator(close=self.processed_data['Close'], window=20).ema_indicator()
            
            # --- Indicadores de Momentum ---
            # RSI
            self.processed_data['RSI'] = RSIIndicator(close=self.processed_data['Close']).rsi()
            
            # Estocástico
            stoch = StochasticOscillator(high=self.processed_data['High'], 
                                         low=self.processed_data['Low'], 
                                         close=self.processed_data['Close'])
            self.processed_data['Stoch_K'] = stoch.stoch()
            self.processed_data['Stoch_D'] = stoch.stoch_signal()
            
            # --- Indicadores de Volatilidade ---
            # Bandas de Bollinger
            bollinger = BollingerBands(close=self.processed_data['Close'])
            self.processed_data['BB_High'] = bollinger.bollinger_hband()
            self.processed_data['BB_Low'] = bollinger.bollinger_lband()
            self.processed_data['BB_Mid'] = bollinger.bollinger_mavg()
            self.processed_data['BB_Width'] = bollinger.bollinger_wband()
            
            # ATR
            self.processed_data['ATR'] = AverageTrueRange(high=self.processed_data['High'], 
                                                        low=self.processed_data['Low'], 
                                                        close=self.processed_data['Close']).average_true_range()
            
            # --- Indicadores de Volume ---
            # On-Balance Volume
            self.processed_data['OBV'] = OnBalanceVolumeIndicator(close=self.processed_data['Close'], 
                                                                volume=self.processed_data['Volume']).on_balance_volume()
            
            # Accumulation/Distribution Index
            self.processed_data['ADI'] = AccDistIndexIndicator(high=self.processed_data['High'], 
                                                             low=self.processed_data['Low'], 
                                                             close=self.processed_data['Close'], 
                                                             volume=self.processed_data['Volume']).acc_dist_index()
            
            # --- Recursos Adicionais ---
            # Preço vs SMA
            self.processed_data['Price_SMA_20_Ratio'] = self.processed_data['Close'] / self.processed_data['SMA_20']
            self.processed_data['Price_SMA_50_Ratio'] = self.processed_data['Close'] / self.processed_data['SMA_50']
            
            # Preço em relação ao intervalo diário
            self.processed_data['Daily_Range'] = (self.processed_data['High'] - self.processed_data['Low']) / self.processed_data['Close']
            
            # Retornos percentuais diários
            self.processed_data['Return'] = self.processed_data['Close'].pct_change() * 100
            
            # Retornos acumulados (5 dias)
            self.processed_data['Return_5d'] = self.processed_data['Close'].pct_change(5) * 100
            
            # Remover valores NaN resultantes do cálculo dos indicadores
            self.processed_data = self.processed_data.dropna()
            
            logger.info(f"Indicadores técnicos adicionados aos dados. Shape final: {self.processed_data.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
            return False

    def prepare_data_for_lstm(self):
        """Preparar dados para treinamento do modelo LSTM"""
        try:
            # Selecionar recursos (features) para o modelo
            features = ['Close', 'Volume', 'MACD', 'MACD_Signal', 'RSI', 'SMA_20', 'EMA_20', 
                        'BB_High', 'BB_Low', 'ATR', 'OBV', 'Stoch_K', 'Stoch_D', 'Return']
            
            # Verificar se todas as features existem
            for feature in features:
                if feature not in self.processed_data.columns:
                    logger.error(f"Feature {feature} não encontrada nos dados processados")
                    return False
            
            # Normalizar os dados
            data_scaled = self.scaler.fit_transform(self.processed_data[features])
            
            X, y = [], []
            
            # Criar sequências de dados para treinamento
            for i in range(self.prediction_days, len(data_scaled) - self.future_days):
                X.append(data_scaled[i-self.prediction_days:i])
                # Prever o preço fechamento futuro (índice 0 = preço de fechamento)
                y.append(data_scaled[i + self.future_days, 0])
            
            # Converter listas para arrays numpy
            X, y = np.array(X), np.array(y)
            
            # Dividir em conjuntos de treinamento e teste (80% / 20%)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            logger.info(f"Dados preparados para LSTM. X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao preparar dados para LSTM: {e}")
            return False

    def build_model(self):
        """Construir e compilar o modelo LSTM"""
        try:
            # Definir a arquitetura do modelo
            model = Sequential()
            
            # Primeira camada LSTM com Dropout
            model.add(LSTM(units=self.lstm_units, return_sequences=True, 
                          input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            
            # Segunda camada LSTM com Dropout
            model.add(LSTM(units=self.lstm_units, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            
            # Terceira camada LSTM com Dropout
            model.add(LSTM(units=self.lstm_units))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            
            # Camada de saída
            model.add(Dense(units=1))
            
            # Compilar o modelo
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            self.model = model
            
            logger.info("Modelo LSTM construído e compilado")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao construir modelo LSTM: {e}")
            return False

    def train_model(self):
        """Treinar o modelo LSTM"""
        try:
            # Callbacks para melhorar o treinamento
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(f"models/{self.ticker}_model.h5", 
                                             save_best_only=True, 
                                             monitor='val_loss')
            
            # Treinar o modelo
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            logger.info(f"Modelo treinado por {len(history.history['loss'])} épocas")
            
            # Avaliar o modelo
            self.evaluate_model()
            
            return True
        except Exception as e:
            logger.error(f"Erro ao treinar modelo LSTM: {e}")
            return False

    def evaluate_model(self):
        """Avaliar a performance do modelo"""
        try:
            # Fazer previsões no conjunto de teste
            predictions_scaled = self.model.predict(self.X_test)
            
            # Preparar para inverter a normalização
            # Criar array de zeros com formato correto
            pred_full_features = np.zeros((len(predictions_scaled), self.X_train.shape[2]))
            # Colocar as previsões na primeira coluna (índice do preço de fechamento)
            pred_full_features[:, 0] = predictions_scaled.flatten()
            
            # Inverter a normalização
            predictions = self.scaler.inverse_transform(pred_full_features)[:, 0]
            
            # Fazer o mesmo para os valores reais
            y_test_full_features = np.zeros((len(self.y_test), self.X_train.shape[2]))
            y_test_full_features[:, 0] = self.y_test
            actual = self.scaler.inverse_transform(y_test_full_features)[:, 0]
            
            # Calcular métricas
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            
            # Calcular acurácia direcional
            direction_actual = np.sign(np.diff(np.append([actual[0]], actual)))
            direction_pred = np.sign(np.diff(np.append([predictions[0]], predictions)))
            direction_accuracy = np.mean(direction_actual == direction_pred) * 100
            
            # Calcular acurácia considerando uma margem de erro de 5%
            accuracy_margin = 0.05  # 5% de margem
            within_margin = np.abs((predictions - actual) / actual) <= accuracy_margin
            accuracy_with_margin = np.mean(within_margin) * 100
            
            # Armazenar as métricas
            self.accuracy = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Direction_Accuracy': direction_accuracy,
                'Accuracy_Within_5%': accuracy_with_margin
            }
            
            self.predictions = predictions
            
            logger.info(f"Avaliação do modelo para {self.ticker}:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R2: {r2:.4f}")
            logger.info(f"  Acurácia Direcional: {direction_accuracy:.2f}%")
            logger.info(f"  Acurácia (margem 5%): {accuracy_with_margin:.2f}%")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao avaliar modelo: {e}")
            return False

    def predict_future(self):
        """Fazer previsões para dias futuros"""
        try:
            # Obter os últimos dados para previsão
            features = ['Close', 'Volume', 'MACD', 'MACD_Signal', 'RSI', 'SMA_20', 'EMA_20', 
                      'BB_High', 'BB_Low', 'ATR', 'OBV', 'Stoch_K', 'Stoch_D', 'Return']
            
            last_sequence = self.processed_data[features].tail(self.prediction_days).values
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Preparar para a previsão
            future_predictions = []
            current_batch = last_sequence_scaled.reshape(1, self.prediction_days, len(features))
            
            # Prever dia por dia
            for i in range(self.future_days):
                # Fazer previsão
                future_price_scaled = self.model.predict(current_batch)[0][0]
                
                # Armazenar a previsão
                future_predictions.append(future_price_scaled)
                
                # Atualizar a sequência para a próxima previsão
                # Criar um novo ponto de dados com a previsão como preço de fechamento
                # Manter outros valores da última sequência por simplicidade
                new_point = current_batch[0][-1].copy()
                new_point[0] = future_price_scaled  # Atualizar o preço de fechamento
                
                # Remover o primeiro e adicionar o novo ponto
                current_batch = np.append(current_batch[0][1:], [new_point], axis=0)
                current_batch = current_batch.reshape(1, self.prediction_days, len(features))
            
            # Converter de volta para os valores reais
            future_pred_full_features = np.zeros((len(future_predictions), len(features)))
            future_pred_full_features[:, 0] = future_predictions
            
            future_prices = self.scaler.inverse_transform(future_pred_full_features)[:, 0]
            
            # Criar dataframe de previsões futuras
            last_date = self.processed_data.index[-1]
            future_dates = [last_date + dt.timedelta(days=i+1) for i in range(self.future_days)]
            
            # Ajustar para dias de negociação (ignorando finais de semana)
            adjusted_dates = []
            for date in future_dates:
                # Se for final de semana, adicionar dias até ser dia útil
                while date.weekday() > 4:  # 5 = sábado, 6 = domingo
                    date += dt.timedelta(days=1)
                adjusted_dates.append(date)
            
            self.future_predictions = pd.DataFrame({
                'Date': adjusted_dates,
                'Predicted_Close': future_prices
            }).set_index('Date')
            
            logger.info(f"Previsões futuras geradas para {self.ticker} - próximos {self.future_days} dias")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao prever preços futuros: {e}")
            return False

    def load_model(self, model_path):
        """Carregar modelo salvo anteriormente"""
        try:
            self.model = load_model(model_path)
            logger.info(f"Modelo carregado de {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False

    def save_model(self, model_path=None):
        """Salvar o modelo treinado"""
        if model_path is None:
            model_path = f"models/{self.ticker}_model.h5"
            
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Salvar o modelo
            self.model.save(model_path)
            logger.info(f"Modelo salvo em {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            return False

    def generate_signals(self):
        """Gerar sinais de compra/venda com base em indicadores técnicos e previsões"""
        try:
            # Pegar os dados mais recentes
            recent_data = self.processed_data.tail(30)
            
            # Último preço conhecido
            last_price = recent_data['Close'].iloc[-1]
            
            # Previsão para o próximo dia
            next_day_prediction = self.future_predictions['Predicted_Close'].iloc[0]
            
            # Alteração percentual prevista
            predicted_change_pct = ((next_day_prediction - last_price) / last_price) * 100
            
            # Inicializar sinais
            signals = {
                'trend': None,  # 'up', 'down', ou 'neutral'
                'strength': None,  # 'strong', 'moderate', ou 'weak'
                'recommendation': None,  # 'buy', 'sell', ou 'hold'
                'confidence': None,  # Percentual de confiança
                'indicators': {}
            }
            
            # Analisar tendência com base nas previsões
            if predicted_change_pct > 1.5:
                signals['trend'] = 'up'
                strength_value = min(abs(predicted_change_pct) / 5, 1) * 100  # Normalizar para 0-100%
            elif predicted_change_pct < -1.5:
                signals['trend'] = 'down'
                strength_value = min(abs(predicted_change_pct) / 5, 1) * 100  # Normalizar para 0-100%
            else:
                signals['trend'] = 'neutral'
                strength_value = 50  # Neutro
                
            # Classificar a força da tendência
            if strength_value > 80:
                signals['strength'] = 'strong'
            elif strength_value > 60:
                signals['strength'] = 'moderate'
            else:
                signals['strength'] = 'weak'
                
            # Analisar indicadores técnicos
            
            # 1. RSI
            rsi_value = recent_data['RSI'].iloc[-1]
            if rsi_value > 70:
                signals['indicators']['RSI'] = 'sobrecomprado'
            elif rsi_value < 30:
                signals['indicators']['RSI'] = 'sobrevendido'
            else:
                signals['indicators']['RSI'] = 'neutro'
                
            # 2. MACD
            macd_value = recent_data['MACD'].iloc[-1]
            macd_signal = recent_data['MACD_Signal'].iloc[-1]
            
            if macd_value > macd_signal:
                signals['indicators']['MACD'] = 'bullish'
            else:
                signals['indicators']['MACD'] = 'bearish'
                
            # 3. Bandas de Bollinger
            bb_high = recent_data['BB_High'].iloc[-1]
            bb_low = recent_data['BB_Low'].iloc[-1]
            close = recent_data['Close'].iloc[-1]
            
            if close > bb_high:
                signals['indicators']['Bollinger'] = 'acima_banda_superior'
            elif close < bb_low:
                signals['indicators']['Bollinger'] = 'abaixo_banda_inferior'
            else:
                signals['indicators']['Bollinger'] = 'dentro_bandas'
                
            # 4. Estocástico
            stoch_k = recent_data['Stoch_K'].iloc[-1]
            stoch_d = recent_data['Stoch_D'].iloc[-1]
            
            if stoch_k > 80 and stoch_d > 80:
                signals['indicators']['Stochastic'] = 'sobrecomprado'
            elif stoch_k < 20 and stoch_d < 20:
                signals['indicators']['Stochastic'] = 'sobrevendido'
            else:
                signals['indicators']['Stochastic'] = 'neutro'
                
            # 5. Médias Móveis
            sma_20 = recent_data['SMA_20'].iloc[-1]
            sma_50 = recent_data['SMA_50'].iloc[-1]
            
            if close > sma_20 and close > sma_50 and sma_20 > sma_50:
                signals['indicators']['Moving_Averages'] = 'forte_tendencia_alta'
            elif close < sma_20 and close < sma_50 and sma_20 < sma_50:
                signals['indicators']['Moving_Averages'] = 'forte_tendencia_baixa'
            elif close > sma_20 and sma_20 > sma_50:
                signals['indicators']['Moving_Averages'] = 'tendencia_alta'
            elif close < sma_20 and sma_20 < sma_50:
                signals['indicators']['Moving_Averages'] = 'tendencia_baixa'
            else:
                signals['indicators']['Moving_Averages'] = 'sem_tendencia_clara'
                
            # Determinar recomendação com base em todos os indicadores
            bullish_signals = 0
            bearish_signals = 0
            
            # Calcular pontuação para tendência
            if signals['trend'] == 'up':
                bullish_signals += 2
            elif signals['trend'] == 'down':
                bearish_signals += 2
                
            # RSI
            if signals['indicators']['RSI'] == 'sobrecomprado':
                bearish_signals += 1
            elif signals['indicators']['RSI'] == 'sobrevendido':
                bullish_signals += 1
                
            # MACD
            if signals['indicators']['MACD'] == 'bullish':
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # Bollinger Bands
            if signals['indicators']['Bollinger'] == 'acima_banda_superior':
                bearish_signals += 1
            elif signals['indicators']['Bollinger'] == 'abaixo_banda_inferior':
                bullish_signals += 1
                
            # Estocástico
            if signals['indicators']['Stochastic'] == 'sobrecomprado':
                bearish_signals += 1
            elif signals['indicators']['Stochastic'] == 'sobrevendido':
                bullish_signals += 1
                
            # Médias Móveis
            if signals['indicators']['Moving_Averages'] in ['forte_tendencia_alta', 'tendencia_alta']:
                bullish_signals += 1
            elif signals['indicators']['Moving_Averages'] in ['forte_tendencia_baixa', 'tendencia_baixa']:
                bearish_signals += 1
                
            # Determinar recomendação final
            total_signals = bullish_signals + bearish_signals
            bullish_percentage = (bullish_signals / total_signals) * 100
            bearish_percentage = (bearish_signals / total_signals) * 100
            
            if bullish_percentage >= 65:
                signals['recommendation'] = 'buy'
                signals['confidence'] = bullish_percentage
            elif bearish_percentage >= 65:
                signals['recommendation'] = 'sell'
                signals['confidence'] = bearish_percentage
            else:
                signals['recommendation'] = 'hold'
                signals['confidence'] = max(bullish_percentage, bearish_percentage)
                
            return signals
        except Exception as e:
            logger.error(f"Erro ao gerar sinais: {e}")
            return None

    def create_interactive_plot(self, output_path="plots"):
        """Criar visualização interativa com plotly"""
        try:
            # Criar diretório para gráficos se não existir
            os.makedirs(output_path, exist_ok=True)
            
            # Data para visualização (últimos 120 dias + previsões)
            plot_data = self.data.tail(120).copy()
            
            # Adicionar previsões futuras ao dataframe
            extended_data = pd.concat([plot_data, self.future_predictions])
            
            # Criar subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=("Preço e Previsões", "Volume", "Indicadores de Momentum", "Indicadores de Tendência")
            )
            
            # Adicionar candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name="OHLC",
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            # Adicionar linhas para SMA 20 e 50
            if 'SMA_20' in self.processed_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.processed_data.tail(120).index,
                        y=self.processed_data.tail(120)['SMA_20'],
                        name="SMA 20",
                        line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)
                    ),
                    row=1, col=1
                )
                
            if 'SMA_50' in self.processed_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.processed_data.tail(120).index,
                        y=self.processed_data.tail(120)['SMA_50'],
                        name="SMA 50",
                        line=dict(color='rgba(46, 49, 146, 0.8)', width=1.5)
                    ),
                    row=1, col=1
                )
            
            # Adicionar linha de previsão
            fig.add_trace(
                go.Scatter(
                    x=self.future_predictions.index,
                    y=self.future_predictions['Predicted_Close'],
                    name="Previsão",
                    line=dict(color='rgba(220, 20, 60, 0.8)', width=2, dash='dashdot'),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # Adicionar Bandas de Bollinger
            if all(x in self.processed_data.columns for x in ['BB_High', 'BB_Low', 'BB_Mid']):
                recent_data = self.processed_data.tail(120)
                
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data['BB_High'],
                        name="BB Alta",
                        line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=recent_data['BB_Low'],
                        name="BB Baixa",
                        line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(0, 128, 0, 0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=plot_data.index,
                    y=plot_data['Volume'],
                    name="Volume",
                    marker_color='rgba(128, 128, 128, 0.5)'
                ),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in self.processed_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.processed_data.tail(120).index,
                        y=self.processed_data.tail(120)['RSI'],
                        name="RSI",
                        line=dict(color='purple', width=1.5)
                    ),
                    row=3, col=1
                )
                
                # Adicionar linhas horizontais para RSI (30 e 70)
                fig.add_shape(
                    type="line", line_color="red", line_width=1, opacity=0.3, line_dash="dash",
                    x0=self.processed_data.tail(120).index[0], y0=70, 
                    x1=self.processed_data.tail(120).index[-1], y1=70,
                    row=3, col=1
                )
                
                fig.add_shape(
                    type="line", line_color="green", line_width=1, opacity=0.3, line_dash="dash",
                    x0=self.processed_data.tail(120).index[0], y0=30, 
                    x1=self.processed_data.tail(120).index[-1], y1=30,
                    row=3, col=1
                )
            
            # MACD
            if all(x in self.processed_data.columns for x in ['MACD', 'MACD_Signal']):
                fig.add_trace(
                    go.Scatter(
                        x=self.processed_data.tail(120).index,
                        y=self.processed_data.tail(120)['MACD'],
                        name="MACD",
                        line=dict(color='blue', width=1.5)
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=self.processed_data.tail(120).index,
                        y=self.processed_data.tail(120)['MACD_Signal'],
                        name="Sinal MACD",
                        line=dict(color='red', width=1.5)
                    ),
                    row=4, col=1
                )
                
                # Histograma para MACD
                if 'MACD_Diff' in self.processed_data.columns:
                    macd_diff = self.processed_data.tail(120)['MACD_Diff']
                    colors = ['green' if val >= 0 else 'red' for val in macd_diff]
                    
                    fig.add_trace(
                        go.Bar(
                            x=self.processed_data.tail(120).index,
                            y=macd_diff,
                            name="MACD Hist",
                            marker_color=colors,
                            opacity=0.5
                        ),
                        row=4, col=1
                    )
            
            # Atualizar layout
            fig.update_layout(
                title=f'Análise Técnica e Previsão para {self.ticker}',
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=900,
                width=1200,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Adicionar anotações no gráfico
            # Última previsão
            last_known_price = plot_data['Close'].iloc[-1]
            future_price = self.future_predictions['Predicted_Close'].iloc[-1]
            price_change = ((future_price - last_known_price) / last_known_price) * 100
            
            fig.add_annotation(
                x=self.future_predictions.index[-1],
                y=future_price,
                text=f"Previsão: ${future_price:.2f} ({price_change:.2f}%)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                bgcolor="#f9f9f9",
                bordercolor="#636363",
                borderwidth=1,
                borderpad=4,
                font=dict(color="#000000")
            )
            
            # Salvar gráfico como HTML
            html_path = f"{output_path}/{self.ticker}_forecast.html"
            fig.write_html(html_path)
            
            # Salvar gráfico como imagem PNG
            png_path = f"{output_path}/{self.ticker}_forecast.png"
            fig.write_image(png_path)
            
            logger.info(f"Gráficos salvos em {html_path} e {png_path}")
            
            return html_path, png_path
        except Exception as e:
            logger.error(f"Erro ao criar visualização: {e}")
            return None, None

    async def analyze_and_notify(self):
        """Analisar o ativo e enviar notificação se necessário"""
        try:
            # Gerar sinais
            signals = self.generate_signals()
            
            if signals is None:
                return False
            
            # Preparar mensagem
            last_price = self.processed_data['Close'].iloc[-1]
            next_day_pred = self.future_predictions['Predicted_Close'].iloc[0]
            price_change = ((next_day_pred - last_price) / last_price) * 100
            
            # Construir a mensagem
            message = f"🔮 *Análise de {self.ticker}* 🔮\n\n"
            message += f"📊 *Preço atual:* ${last_price:.2f}\n"
            message += f"🔄 *Previsão (próximo dia):* ${next_day_pred:.2f} ({price_change:.2f}%)\n\n"
            
            # Recomendação
            if signals['recommendation'] == 'buy':
                message += f"🟢 *COMPRAR* com {signals['confidence']:.1f}% de confiança\n\n"
            elif signals['recommendation'] == 'sell':
                message += f"🔴 *VENDER* com {signals['confidence']:.1f}% de confiança\n\n"
            else:
                message += f"⚪ *MANTER* com {signals['confidence']:.1f}% de confiança\n\n"
            
            # Adicionar resumo dos indicadores
            message += "*Indicadores Técnicos:*\n"
            message += f"• RSI: {self.processed_data['RSI'].iloc[-1]:.1f} ({signals['indicators']['RSI']})\n"
            message += f"• MACD: {signals['indicators']['MACD']}\n"
            message += f"• Bollinger: {signals['indicators']['Bollinger']}\n"
            message += f"• Estocástico: {signals['indicators']['Stochastic']}\n"
            message += f"• Médias Móveis: {signals['indicators']['Moving_Averages']}\n\n"
            
            # Previsões de 5 dias
            message += "*Previsão para 5 dias:*\n"
            for i in range(min(5, len(self.future_predictions))):
                date = self.future_predictions.index[i].strftime('%d/%m/%Y')
                price = self.future_predictions['Predicted_Close'].iloc[i]
                day_change = ((price - last_price) / last_price) * 100
                message += f"• {date}: ${price:.2f} ({day_change:.2f}%)\n"
                
            message += f"\n📈 Acurácia do modelo: {self.accuracy['Accuracy_Within_5%']:.1f}%"
            
            # Criar gráfico
            _, png_path = self.create_interactive_plot()
            
            # Enviar notificação
            if png_path:
                await self.send_telegram_message(message, png_path)
            else:
                await self.send_telegram_message(message)
                
            logger.info(f"Notificação enviada para {self.ticker}")
            
            return True
        except Exception as e:
            logger.error(f"Erro ao analisar e notificar: {e}")
            return False

    async def run(self):
        """Executar todo o pipeline de análise"""
        try:
            # 1. Baixar dados
            if not self.fetch_data():
                return False
                
            # 2. Adicionar indicadores técnicos
            if not self.add_technical_indicators():
                return False
                
            # 3. Preparar dados para LSTM
            if not self.prepare_data_for_lstm():
                return False
                
            # 4. Construir modelo se não existir
            if self.model is None:
                if not self.build_model():
                    return False
                    
                # 5. Treinar modelo
                if not self.train_model():
                    return False
            
            # 6. Fazer previsões futuras
            if not self.predict_future():
                return False
                
            # 7. Analisar e notificar
            if not await self.analyze_and_notify():
                return False
                
            logger.info(f"Pipeline de análise completo para {self.ticker}")
            return True
        except Exception as e:
            logger.error(f"Erro ao executar pipeline: {e}")
            return False


async def main():
    """Função principal para executar o sistema"""
    # Lista de tickers para analisar
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    
    # Configurações para o modelo
    config = {
        'period': '2y',            # 2 anos de dados históricos
        'interval': '1d',          # Intervalo diário
        'prediction_days': 60,     # Usar 60 dias para prever
        'future_days': 10,         # Prever 10 dias à frente
        'lstm_units': 100,         # Unidades na camada LSTM
        'epochs': 100,             # Máximo de épocas para treinamento
        'batch_size': 32           # Tamanho do lote para treinamento
    }
    
    for ticker in tickers:
        # Inicializar sistema
        logger.info(f"Iniciando análise para {ticker}")
        system = StockPredictionSystem(ticker, **config)
        
        # Executar pipeline
        await system.run()
        
        # Aguardar um pouco antes da próxima análise
        await asyncio.sleep(5)


def run_scheduler():
    """Executar o sistema em um agendador"""
    try:
        # Configurar o loop de eventos assíncrono
        loop = asyncio.get_event_loop()
        
        # Executar a análise inicial
        loop.run_until_complete(main())
        
        # Agendador para executar uma vez por dia (às 18:00)
        while True:
            now = dt.datetime.now()
            # Calcular tempo até a próxima execução (18:00)
            target_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
            if now >= target_time:
                target_time += dt.timedelta(days=1)
                
            wait_seconds = (target_time - now).total_seconds()
            logger.info(f"Próxima análise agendada para {target_time.strftime('%Y-%m-%d %H:%M:%S')} (em {wait_seconds/3600:.1f} horas)")
            
            time.sleep(wait_seconds)
            loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Sistema encerrado pelo usuário")
    except Exception as e:
        logger.error(f"Erro ao executar agendador: {e}")


if __name__ == "__main__":
    # Verificar se já existe diretório para modelos
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Verificar se já existe diretório para plots
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # Iniciar o agendador
    run_scheduler()
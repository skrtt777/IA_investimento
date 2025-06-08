import sys
import types
import pandas as pd
import pytest

# Stub modules that may not be installed
missing_modules = [
    'telegram',
    'plotly',
    'plotly.graph_objects',
    'plotly.express',
    'plotly.subplots',
    'tensorflow',
    'tensorflow.keras',
    'tensorflow.keras.models',
    'tensorflow.keras.layers',
    'tensorflow.keras.callbacks',
    'keras_tuner',
    'ta',
    'ta.trend',
    'ta.momentum',
    'ta.volatility',
    'ta.volume',
    'matplotlib',
    'matplotlib.pyplot',
]

for name in missing_modules:
    if name not in sys.modules:
        module = types.ModuleType(name)
        sys.modules[name] = module

# Provide minimal attributes for submodules used during import
sys.modules['plotly.subplots'].make_subplots = lambda *a, **k: None

class DummyBot:
    def __init__(self, *a, **k):
        pass

sys.modules['telegram'].Bot = DummyBot

# Minimal tensorflow stubs
keras_models = types.ModuleType('models')
keras_models.Sequential = object
keras_models.load_model = lambda *a, **k: None
sys.modules['tensorflow.keras.models'] = keras_models

keras_layers = types.ModuleType('layers')
keras_layers.Dense = object
keras_layers.LSTM = object
keras_layers.Dropout = object
keras_layers.BatchNormalization = object
sys.modules['tensorflow.keras.layers'] = keras_layers

keras_callbacks = types.ModuleType('callbacks')
keras_callbacks.EarlyStopping = object
keras_callbacks.ModelCheckpoint = object
sys.modules['tensorflow.keras.callbacks'] = keras_callbacks

# ta library stubs
ta_trend = types.ModuleType('ta.trend')
ta_trend.MACD = object
ta_trend.SMAIndicator = object
ta_trend.EMAIndicator = object
sys.modules['ta.trend'] = ta_trend

ta_momentum = types.ModuleType('ta.momentum')
ta_momentum.RSIIndicator = object
ta_momentum.StochasticOscillator = object
sys.modules['ta.momentum'] = ta_momentum

ta_volatility = types.ModuleType('ta.volatility')
ta_volatility.BollingerBands = object
ta_volatility.AverageTrueRange = object
sys.modules['ta.volatility'] = ta_volatility

ta_volume = types.ModuleType('ta.volume')
ta_volume.OnBalanceVolumeIndicator = object
ta_volume.AccDistIndexIndicator = object
sys.modules['ta.volume'] = ta_volume

sys.modules['ta'] = types.ModuleType('ta')

from IA_investimento import StockPredictionSystem


def test_fetch_data_returns_false_when_empty(monkeypatch):
    monkeypatch.setattr('yfinance.download', lambda *a, **k: pd.DataFrame())
    system = StockPredictionSystem('DUMMY')
    assert system.fetch_data() is False

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from backtesting import Backtest, Strategy

# Carregar e pré-processar os dados da bolsa de valores
# Substitua pelo seu conjunto de dados
dados = pd.read_csv('dados_petroleo_2021.csv')
dados = dados['Close']  # Considere apenas o preço de fechamento

# Normalizar os dados
dados_normalizados = (dados - dados.mean()) / dados.std()

# Hiperparâmetros da GAN
dimensao_latente = 10
epocas = 150
tamanho_lote = 128

# Gerador
gerador = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(dimensao_latente,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='linear')
])

# Discriminador
discriminador = tf.keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Função de perda e otimizadores
funcao_perda = tf.keras.losses.BinaryCrossentropy()
otimizador_gerador = tf.keras.optimizers.Adam(learning_rate=0.0001)
otimizador_discriminador = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Função de treinamento da GAN
@tf.function
def treinar_gan(dados_reais):
    tamanho_lote = dados_reais.shape[0]
    vetor_latente = tf.random.normal([tamanho_lote, dimensao_latente])

    with tf.GradientTape() as gerador_tape, tf.GradientTape() as discriminador_tape:
        dados_gerados = gerador(vetor_latente)

        resultado_dados_reais = discriminador(dados_reais)
        resultado_dados_gerados = discriminador(dados_gerados)

        perda_discriminador = funcao_perda(tf.ones_like(resultado_dados_reais), resultado_dados_reais) + \
            funcao_perda(tf.zeros_like(resultado_dados_gerados), resultado_dados_gerados)
        perda_gerador = funcao_perda(tf.ones_like(resultado_dados_gerados), resultado_dados_gerados)

    gradientes_gerador = gerador_tape.gradient(perda_gerador, gerador.trainable_variables)
    gradientes_discriminador = discriminador_tape.gradient(perda_discriminador, discriminador.trainable_variables)

    otimizador_gerador.apply_gradients(zip(gradientes_gerador, gerador.trainable_variables))
    otimizador_discriminador.apply_gradients(zip(gradientes_discriminador, discriminador.trainable_variables))

# Treinamento da GAN
for epoca in range(epocas):
    indice_aleatorio = np.random.randint(0, len(dados_normalizados) - tamanho_lote)
    dados_reais = dados_normalizados[indice_aleatorio:indice_aleatorio + tamanho_lote].values.reshape(-1, 1)

    treinar_gan(dados_reais)

    if epoca % 100 == 0:
        print(f'Época {epoca}/{epocas}')

# Gerar previsões futuras
vetor_latente = tf.random.normal([tamanho_lote, dimensao_latente])
previsoes_futuras = gerador(vetor_latente)

# Desnormalizar as previsões
previsoes_futuras = previsoes_futuras.numpy().reshape(-1) * dados.std() + dados.mean()

print(previsoes_futuras)

# Gerar datas para os dados futuros
ultima_data = pd.to_datetime(dados.index[-1])
datas_futuras = pd.date_range(start=ultima_data + pd.DateOffset(days=1), periods=len(previsoes_futuras)).normalize()

# Criar DataFrame para backtesting
dados_backtesting = pd.DataFrame({'Date': datas_futuras, 'Close': previsoes_futuras})
dados_backtesting.set_index('Date', inplace=True)
dados_backtesting['Open'] = dados_backtesting['High'] = dados_backtesting['Low'] = dados_backtesting['Close']

# Definir a estratégia de backtesting (cruzamento de médias)
class MovingAverageCrossStrategy(Strategy):
    def init(self):
        self.fast_ma_window = 50  # Janela da média móvel rápida
        self.slow_ma_window = 200  # Janela da média móvel lenta
        self.buy_signal_triggered = False

    def next(self):
        close_prices = self.data.Close
        # Cálculo da média móvel rápida
        fast_ma = close_prices[-self.fast_ma_window:].mean()
        # Cálculo da média móvel lenta
        slow_ma = close_prices[-self.slow_ma_window:].mean()

        if fast_ma > slow_ma and not self.buy_signal_triggered:
            self.buy()
            self.buy_signal_triggered = True
        elif fast_ma < slow_ma and self.buy_signal_triggered:
            self.sell()
            self.buy_signal_triggered = False

# Executar o backtesting
bt = Backtest(dados_backtesting, MovingAverageCrossStrategy)
resultado = bt.run()

print(resultado)

import numpy as np
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt

def embaralhamento (sinal_entrada, fs):
    t = np.linspace(0, len(sinal_entrada) / fs, num=len(sinal_entrada))

    #1)Primeira modulação:
    f_c1 = 20000 #frequencia da portadora 1
    c1 = 2 * np.cos(2 * np.pi * f_c1 * t) #vetor portadora
    x1 = sinal_entrada * c1

    #2)Filtragem passa altas:
    f_fpa = f_c1 #frequência de corte
    ordem = 8
    #(normalização da frequência para desing do filtro)
    nyquist = fs/2
    normal_cutoff_fpa = f_fpa / nyquist
    #filtro Butterworth e obtém os coeficientes 'sos' (second-order sections)
    sos_pa = butter(8, normal_cutoff_fpa, btype='high', analog=False, output='sos')
    x2 = sosfilt(sos_pa, x1)

    #3)Segunda Modulação:
    f_c2 = 25000
    c2 = 2 * np.cos( 2 * np.pi * f_c2 * t)
    x3 = x2 * c2

    #4) Filtragem passa baixas:
    f_fpb = 20000
    normal_cutoff_fpb = f_fpb / nyquist
    sos_pb = butter(8, normal_cutoff_fpb, btype='low', analog=False, output='sos')
    sinal_saida = sosfilt(sos_pb, x3)

    return sinal_saida, x1, x2, x3

def plot_espectro(sinal, fs, titulo, ax):
    n = len(sinal)
    # Calcula a FFT (Transformada Rápida de Fourier)
    yf = np.fft.fft(sinal)
    # Calcula as frequências correspondentes
    xf = np.fft.fftfreq(n, 1 / fs)
    
    # Usa fftshift para centralizar o espectro em 0 Hz
    ax.plot(np.fft.fftshift(xf) / 1000, np.abs(np.fft.fftshift(yf)))
    ax.set_title(titulo)
    ax.set_xlabel('Frequência (kHz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    ax.set_xlim(-fs/2/1000, fs/2/1000) # Mostra o espectro completo
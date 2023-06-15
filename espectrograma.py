import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Cargar el archivo de audio
audio_path = 'guitaracoustic-gmaj7-chord.wav'

# y=representa las amplitudes de las muestras de audio en formato numérico
#sr=representa la tasa de muestreo
y, sr = librosa.load(audio_path)

# Calcular el espectrograma utilizando la transformada de Fourier de corto tiempo (STFT)
D = librosa.stft(y)

# Convertir el espectrograma a escala logarítmica
spectrogram = librosa.amplitude_to_db(abs(D), ref=np.max)


chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Graficar el espectrograma en el primer subplot
ax1.set_title('Espectrograma')
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax1)
#plt.colorbar(format='%+2.0f dB')
ax1.set_ylabel('Frecuencia')

# Graficar el cromagrama en el segundo subplot
ax2.set_title('Cromagrama de Chroma CENS')
librosa.display.specshow(chroma_cens, sr=sr, x_axis='time', y_axis='chroma',  ax=ax2)
#plt.colorbar()
ax2.set_ylabel('Chroma')

# Ajustar los espacios entre los subplots
plt.tight_layout()

# Mostrar la figura
plt.show()
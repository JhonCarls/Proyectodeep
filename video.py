import yt_dlp
import os

def descargar_video(url, carpeta_salida='descargas'):
    """
    Descarga un video de YouTube usando yt-dlp.
    """
    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Configuración de las opciones de descarga
    ydl_opts = {
        'format': 'best',  # Descarga la mejor calidad combinada (audio + video)
        'outtmpl': f'{carpeta_salida}/%(title)s.%(ext)s', # Plantilla de nombre de archivo
        'noplaylist': True, # Si el link es una lista, solo descarga el video actual
        'quiet': False,     # Muestra el progreso en la consola
    }

    try:
        print(f"Iniciando descarga de: {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"\n¡Descarga completada exitosamente en la carpeta '{carpeta_salida}'!")
        
    except Exception as e:
        print(f"\nOcurrió un error: {e}")

if __name__ == "__main__":
    # Solicitar URL al usuario
    link = input("Introduce la URL del video de YouTube: ")
    descargar_video(link)
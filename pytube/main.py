import shutil
from pytube import YouTube
from pathlib import Path
from time import sleep
from tqdm import tqdm


def progress_function(stream, chunk, bytes_remaining):
    """
    This function is called automatically by Pytube during download.
    - `stream`: pytube stream object being downloaded.
    - `chunk`: segment of media file that was just downloaded.
    - `bytes_remaining`: how many bytes remain to be downloaded.
    """
    #total_size = stream.filesize
    #bytes_downloaded = total_size - bytes_remaining
    progress_bar.update(len(chunk))  # Update the progress bar by the size of the last chunk received.


if __name__ == "__main__":
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.

    to_download = [
        #'https://www.youtube.com/watch?v=KZsk8B_z8pI&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=2',
        #'https://www.youtube.com/watch?v=4EUkNY6Io0U&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=2',
        #'https://www.youtube.com/watch?v=fz9lDRvgHt8&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=3',
        #'https://www.youtube.com/watch?v=5SfdhTIa9iQ&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=4',
        #'https://www.youtube.com/watch?v=RD_vLPCf03w&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=5',
        #'https://www.youtube.com/watch?v=zPQ7KS__aZs&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=6',
        'https://www.youtube.com/watch?v=moESOXirQqE&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=7',
        'https://www.youtube.com/watch?v=GbhV3WitriM&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=8',
        'https://www.youtube.com/watch?v=2SBwUetNhFY&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=9',
        'https://www.youtube.com/watch?v=Tt2AEGASWfo&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=11',
        'https://www.youtube.com/watch?v=LOLNr_hE5mY&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=10',
        'https://www.youtube.com/watch?v=UiO0Y_eFVQY&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=12',
        'https://www.youtube.com/watch?v=B4XEcYUN66o&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=13',
        'https://www.youtube.com/watch?v=qBd4nIfF-Bk&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=14',
        'https://www.youtube.com/watch?v=dOMmsy9CXjs&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=15',
        'https://www.youtube.com/watch?v=uRhCeg_SR9I&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=16',
        'https://www.youtube.com/watch?v=A0ziDICIlMQ&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=17',
        'https://www.youtube.com/watch?v=Z5_wUKbAKC8&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=18',
        'https://www.youtube.com/watch?v=Lkugsh6bQWI&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=19',
        'https://www.youtube.com/watch?v=8HOiMzW8XF0&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=20',
        'https://www.youtube.com/watch?v=yRKw_dIAUH0&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=21',
        'https://www.youtube.com/watch?v=kkN13nEn2WA&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=22',
        'https://www.youtube.com/watch?v=aUjsaHoR1bc&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=23',
        'https://www.youtube.com/watch?v=0aOGndPFnEU&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=24',
        'https://www.youtube.com/watch?v=Hyf7J3AndXQ&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=25',
        'https://www.youtube.com/watch?v=RULYnEYFtAU&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=26',
        'https://www.youtube.com/watch?v=6JN-DJuSFUY&list=PL5EH0ZJ7V0jV7kMYvPcZ7F9oaf_YAlfbI&index=27',
    ]
    for video_url in to_download:
        # yt = YouTube(video)
        yt = YouTube(video_url, on_progress_callback=progress_function)
        # stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first() 
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

        # Initialize progress bar for this video
        progress_bar = tqdm(total=stream.filesize, unit='B', unit_scale=True, desc=f"Downloading {yt.title}", leave=True)
        temp_video_path = stream.download(skip_existing=False)  # Download the video
        progress_bar.close()

        video = Path(temp_video_path)
        new_video_path = outputs_dir / video.name
        shutil.move(str(video), str(new_video_path))

        sleep(5)

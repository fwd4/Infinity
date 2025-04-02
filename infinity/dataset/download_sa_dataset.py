import os
import csv
import json
import tarfile
import shutil
import requests
import pandas as pd
from tqdm import tqdm
from io import StringIO  # Add this import
from urllib.parse import unquote
from pathlib import Path
import time
import datetime
import pytz

class LargeFileDownloader:
    def __init__(self, csv_url, download_dir, log_file='download_log.json'):
        self.csv_url = csv_url
        self.download_dir = Path(download_dir)
        self.images_dir = self.download_dir / 'images'
        self.json_dir = self.download_dir / 'json'
        self.log_file = self.download_dir / log_file
        
        # Create necessary directories
        for dir_path in [self.download_dir, self.images_dir, self.json_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.download_log = self._load_log()
        
    def _process_tar_file(self, tar_path):
        try:
            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(path=self.download_dir / 'temp')
            
            # Move files to appropriate directories
            temp_dir = self.download_dir / 'temp'
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                        shutil.move(str(file_path), str(self.images_dir / file_path.name))
                    elif file_path.suffix.lower() == '.json':
                        shutil.move(str(file_path), str(self.json_dir / file_path.name))
            
            # Clean up
            shutil.rmtree(temp_dir)
            tar_path.unlink()  # Remove the original tar file
            return True
        except Exception as e:
            print(f"Error processing tar file {tar_path}: {str(e)}")
            return False
    
    def _download_file(self, filename, url):
        try:
            local_path = self.download_dir / filename
            # Disable HTTPS proxy for this request
            proxies = {
                'http': None,
                'https': None
            }
            response = requests.get(url, stream=True)#, proxies=proxies)
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024  # 1MB chunks
            
            # Check if starting in daytime Beijing time
            beijing_tz = pytz.timezone('Asia/Shanghai')
            now = datetime.datetime.now(beijing_tz)
            is_daytime = 10 <= now.hour < 22
            max_speed = 10 * 1024 * 1024 if is_daytime else None  # 10MB/s limit only during daytime
            
            with tqdm(total=file_size, unit='iB', unit_scale=True, desc=filename) as pbar:
                with open(local_path, 'wb') as f:
                    start_time = time.time()
                    bytes_downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            size = f.write(chunk)
                            bytes_downloaded += size
                            pbar.update(size)
                            
                            if max_speed:
                                elapsed = time.time() - start_time
                                expected_time = bytes_downloaded / max_speed
                                if expected_time > elapsed:
                                    time.sleep(expected_time - elapsed)
            
            # Process the tar file
            if self._process_tar_file(local_path):
                self.download_log[filename] = {
                    'status': 'completed',
                    'url': url,
                    'size': file_size
                }
                self._save_log()
                return True
            else:
                raise Exception("Failed to process tar file")
            
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            self.download_log[filename] = {
                'status': 'failed',
                'url': url,
                'error': str(e)
            }
            self._save_log()
            return False

    def process_downloads(self):
        files_processed = 0
        
        while True:
            df = self.download_csv_content()
            if df is None:
                return
            
            print(f"Found {len(df)} files to download")
            
            completed = 0
            failed = 0
            for _, row in df.iterrows():
                filename = row['filename']
                url = row['url']
                if self._download_file(filename, url):
                    completed += 1
                else:
                    failed += 1
                
                files_processed += 1
                if files_processed % 10 == 0:
                    print("\nRefreshing CSV content...")
                    break
            
            print(f"\nDownload Summary:")
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
            
            # Only break if we've processed all files in the current DataFrame
            if len(df) <= files_processed:
                break

    def _load_log(self):
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_csv_content(self):
        try:
            # Print completed downloads
            completed_files = [
                filename for filename, info in self.download_log.items() 
                if info['status'] == 'completed'
            ]
            if completed_files:
                print("\nAlready downloaded files:")
                for file in sorted(completed_files):
                    print(f"  - {file}")
                print(f"Total completed: {len(completed_files)}\n")
            
            response = requests.get(self.csv_url)
            response.raise_for_status()
            csv_content = response.text
            df = pd.read_csv(
                StringIO(csv_content),
                delimiter='\t',
                header=0,
                names=['filename', 'url']
            )
            
            # Filter out already completed downloads and sort by filename
            df = df[~df['filename'].isin(completed_files)].sort_values('filename')
            
            return df
        except Exception as e:
            print(f"Error downloading CSV: {str(e)}")
            return None

if __name__ == "__main__":
    # Initialize downloader
    downloader = LargeFileDownloader(
        csv_url="https://scontent-hkg4-2.xx.fbcdn.net/m1/v/t6/An8MNcSV8eixKBYJ2kyw6sfPh-J9U4tH2BV7uPzibNa0pu4uHi6fyXdlbADVO4nfvsWpTwR8B0usCARHTz33cBQNrC0kWZsD1MbBWjw.txt?_nc_oc=AdkwjnepwxUcYUs-dKM05U6rdnu7muAnMXPaJGlyyNgPzevxAS0op2mtSE-XUdiar56VOQ3ymnAz0QZo66ob7nme&ccb=10-5&oh=00_AYEmsj-jf8sAGEUzLJkggq8GsTlFF9gpVabjVyvjdP7suQ&oe=680CBB98&_nc_sid=0fdd51",
        download_dir="/share/public/public_models/SA-1B"
    )
    
    # Start downloading
    downloader.process_downloads()
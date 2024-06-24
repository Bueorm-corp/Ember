import requests
import os

def download_file_from_github(file_name, save_path):
    url = f"https://github.com/BueormLLC/LDM-base/{file_name}"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Error downloading {file_name} from GitHub. Status code: {response.status_code}")

import os
import urllib.request

def download_cwru():
    base_url = "https://engineering.case.edu/sites/default/files/"
    files = {
        'Normal': ['97.mat', '98.mat', '99.mat', '100.mat'],
        'IR007': ['105.mat', '106.mat', '107.mat', '108.mat'],
        'OR007': ['130.mat', '131.mat', '132.mat', '133.mat'],
        'Ball007': ['118.mat', '119.mat', '120.mat', '121.mat']
    }

    print("Downloading CWRU 12k DE dataset...")
    for folder, fnames in files.items():
        folder_path = os.path.join('data', 'cwru', folder)
        os.makedirs(folder_path, exist_ok=True)
        for fname in fnames:
            dest = os.path.join(folder_path, fname)
            if not os.path.exists(dest):
                url = base_url + fname
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"Downloaded {fname} to {folder}")
                except Exception as e:
                    print(f"Failed to download {fname}: {e}")
            else:
                print(f"File {fname} already exists in {folder}")

if __name__ == '__main__':
    download_cwru()

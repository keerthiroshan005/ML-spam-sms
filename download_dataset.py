from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = DATA_DIR / "sms_spam_collection.zip"
output_path = DATA_DIR / "SMSSpamCollection"

print("Downloading UCI SMS Spam Collection...")

urllib.request.urlretrieve(url, zip_path)

print("Download completed.")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print(f"Dataset saved to: {output_path}")
print("Dataset download completed successfully!")
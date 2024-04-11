# Library
import gdown

# Define url and output in order to download compressed models/tokenizers from google drive
url = "https://drive.google.com/uc?id=1PgOSRi5SkGAtoWqK69zNnE8nE6A5GYCg"
output = "./models_and_tokenizers.tar.xz"
gdown.download(url, output)
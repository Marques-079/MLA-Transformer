import requests
from bs4 import BeautifulSoup

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
text = soup.get_text(separator="\n", strip=True)

with open("input.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Webpage text saved to input.txt")

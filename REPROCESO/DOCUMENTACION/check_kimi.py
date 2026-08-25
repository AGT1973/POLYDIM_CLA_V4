import urllib.request
import json
import concurrent.futures

keys = [
    "sk-QThLz6EUVpz8ntvsVjrJpf81icEJkOIHIo8DJlYrWvjEhrzL",
    "sk-mr-d210e331d48dfeedb5a5723d95a0398b504e826130cbf6faa9cc30d0b8ee53a4",
    "sk-b4N0XmPUn5KoGh3N4fucbc3qApGAUENi1630RFyDowYN5ByF",
    "sk-DCHcUkJGGcP7kNIq2JBGB3ZvMNjuX2c5HNy6sRD5spkFqYj0",
    "sk-c8y0ZhoksCxHlN1PSMLwidMKDdPcbA1gdwUONreSsKBo6vfw",
    "sk-XqVWhJ7XacGdESJCta37iYJjYU7eB9RWHpNb3UMCKzuvGHoZ",
    "sk-iVhLgoBxhzOT7lHbz6KNQoUKWLq8XYEwG2ZbqUurbaNdCtSe",
    "sk-2SpS8B2v5RJvKOgc33KhUCP13aylkVmKoP9qLMkEud3wDCv1",
    "sk-edU49HYHSbXIJhl0PVHacvVTetEDYtxHC3Gz2MSAekr7r80T",
    "sk-Ts7JaHtQnJOgoFFsqPDwip1DIPVuRPofiQFobu82tpHRMQu3",
    "sk-dSEAb5jxl8YJpk9RNYebxMZ7nzjbpVsxHvpxbQ0DeHOnWJlv",
    "sk-L2n3zYijzJP8PYJh9ph4Q8x31Vf6zJuDhsZBKa1RHYc4Xeoi",
    "sk-pkYE8Ew4hKFOeEnOyJezsQCds8BkrwmiuNLj8r71QFPtSO2x",
    "sk-3sK2lcOmhOmDjoNRNRG9waUdMBUh2WRu3XUCVmdDvT2xi7z1",
    "sk-QvZFveIqetcjH6ZlJuebNlwBuGoZsfYRjoa25sh1oUUEvDOS",
    "sk-pmc5UTUhwkhYeolHaC1B40ba2c9hwNXMTXAwIvXtnjtOXsWs",
    "sk-4203ecc69f954cda8a6ea9a350ab5211",
    "sk-qY2gnpMVLgvJZGotX7abmbpl1RrW1DuDf3un6vDycWVV1PPl",
    "sk-bfd0Y4uTU4MjXEIppq2kukgRqKNaxP29usObuPHucqVIHoE2",
    "sk-YEg6EAyUW1bbXXnBsVlmWtyppk3e3Uz3F1uJWvtluCqmw80L",
    "sk-TNgkvdRbCtUkK1yyoFWHR5ZsRj4199QRmm7SJIFbR9vNgCcb",
    "sk-jbolacg0dQl3PFEeGk40S8sFIqPsQq7Kf34U372fHSTiK3Gm",
    "sk-BoopT4pS9CQ6gapNcqzn9Ob4lw6OZo5r1XuzBAqQDoZGat5Y",
    "sk-wEAxMoV0WQK23XftkU7m3vO60fTq7oxI7vchZj1WbtKlR0kx",
    "sk-qUO92G5US1YQ2CauUMDZvJZ3dCCOo2sBkICxqeir2GAy1iwj",
    "sk-qw1CQkW2NSlKNukpxGnaHrnCGqi2I9H6glA4Tyrgza8r6jBP",
    "sk-3DTHFzOVtvOtIeMVolFenuDbnpQyN2ONzQTLMvhIaW422Kzl",
    "sk-2RUlSYdx4OfKUWjh2hXsoJ0vsREXMoBbaXN6nKy8ZZdNKBPY",
    "sk-pZkptg5Wz01QpC4zdS1YzRwocJ1OgKRBdVXlc9P29Vcd54cC",
    "sk-uGT1JHpRIdW5bwz0C320jWpC1m5LPWZl4Wm8Gjm7xwZS68a8"
]

def check_key(key):
    url = "https://api.moonshot.cn/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return key, True
    except urllib.error.HTTPError as e:
        return key, False
    except Exception:
        return key, False

valid_keys = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(check_key, keys)
    for key, is_valid in results:
        if is_valid:
            valid_keys.append(key)
            print(f"[OK] {key}")
        else:
            print(f"[FAIL] {key}")

print(f"\nTotal valid keys: {len(valid_keys)}")

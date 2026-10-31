import urllib.request

file_id = "1HbBrdj4rY2rTtHtlva255eh7Ib4ye_5j"
url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

try:
    content = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")
    print("Downloaded Q1_TriBenchmark_Master_Metrics.csv:")
    print(content)
except Exception as e:
    print("Download error:", e)
    # Try alternate URL
    url2 = f"https://drive.google.com/uc?id={file_id}&export=download"
    req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
    try:
        content = urllib.request.urlopen(req2).read().decode("utf-8", errors="ignore")
        print("Downloaded via uc:", content)
    except Exception as e2:
        print("UC error:", e2)

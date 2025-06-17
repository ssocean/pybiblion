
import os
import requests
import gzip
import json
from urllib.parse import urlparse
import os
import requests
from cfg.config import s2api


def download_dataset(dataset_name: str, dest_folder: str = r"D:\s2\papers", max_files: int = 1):
    os.makedirs(dest_folder, exist_ok=True)
    headers = {"x-api-key": s2api}

    url = f"https://api.semanticscholar.org/datasets/v1/release/latest/dataset/{dataset_name}"
    r = requests.get(url, headers=headers).json()

    print(r)
    files = r["files"][:max_files]

    for i, link in enumerate(files):
        # 提取纯净文件名
        fname = os.path.basename(urlparse(link).path)
        dest_path = os.path.join(dest_folder, fname)

        if os.path.exists(dest_path):
            print(f"[{i + 1}] ✅ 已存在，跳过 {fname}")
            continue

        print(f"[{i + 1}] ⬇️ 下载 {fname} ...")
        with requests.get(link, stream=True) as resp:
            resp.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"[{i + 1}] ✅ 完成 {fname}")

    return os.path.join(dest_folder, os.path.basename(files[0]))

def preview_jsonl_gz(file_path: str, num_lines: int = 3):
    print(f"📖 正在预览文件内容：{file_path}")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            print(json.dumps(obj, indent=2))
            if i + 1 >= num_lines:
                break

if __name__ == "__main__":
    dataset = "papers"  # 可修改为 "citations"
    path = download_dataset(dataset)
    preview_jsonl_gz(path)


# import requests
# import json
#
# r = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
#
# print("📦 当前最新版本:", r['release_id'])
# for ds in r['datasets']:
#     print(f"➡️ {ds['name']}: {ds['description']}")

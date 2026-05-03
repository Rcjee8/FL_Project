import json
import os
import sys
import urllib.request

DEST = os.path.join(os.path.dirname(__file__), "static", "blazeface")

BASE = "https://storage.googleapis.com/tfjs-models/savedmodel/blazeface"


def download(url, dest_path):
    print(f"  Downloading {os.path.basename(dest_path)} … ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)

        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            print(f"{size // 1024} KB")
            return True
        else:
            print("FAILED: File not found after download")
            return False

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("\nSetting up BlazeFace for offline use...\n")

    os.makedirs(DEST, exist_ok=True)

    model_path = os.path.join(DEST, "model.json")
    ok = download(f"{BASE}/model.json", model_path)

    if not ok:
        print("\n❌ Failed to download model.json. Check your internet.")
        sys.exit(1)

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip().startswith("<"):
            print("\n❌ Downloaded HTML instead of JSON.")
            print("   The model URL is incorrect or blocked.")
            sys.exit(1)

        manifest = json.loads(content)

    except Exception as e:
        print(f"\n❌ Failed to parse model.json: {e}")
        sys.exit(1)

    shard_names = []
    for group in manifest.get("weightsManifest", []):
        for path in group.get("paths", []):
            shard_names.append(path)

    if not shard_names:
        print("\n❌ No weight shards found — unexpected model format.")
        sys.exit(1)

    print(f"\nFound {len(shard_names)} weight shard(s).\n")

    for name in shard_names:
        shard_path = os.path.join(DEST, name)
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)

        ok = download(f"{BASE}/{name}", shard_path)
        if not ok:
            print(f"\n❌ Failed to download shard: {name}")
            sys.exit(1)

    print("\n✅ BlazeFace setup complete!")
    print(f"📁 Saved to: {DEST}")
    print("🚀 You can now run phone_server.py WITHOUT internet.\n")


if __name__ == "__main__":
    main()
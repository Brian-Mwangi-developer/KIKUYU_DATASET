#!/usr/bin/env python3
"""
Fix Hugging Face Dataset Loading Issue for Kikuyu Dataset
This script addresses the NotImplementedError when loading the dataset
"""

import pandas as pd
import os
from datasets import Dataset, Audio, DatasetDict, load_dataset
from huggingface_hub import HfApi, login, delete_repo, create_repo
import json


def clear_cache_and_retry():
    """
    Clear local cache and try different loading methods
    """
    print("🧹 Clearing local cache...")

    # Clear datasets cache
    try:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()

        # Try to clear the cache
        import shutil
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(os.path.join(
                    cache_dir, "BrianMwangi___kikuyu-transcription-dataset"))
                print("✅ Cleared local cache")
            except:
                print("⚠️  Could not clear cache completely")
    except Exception as e:
        print(f"⚠️  Cache clearing issue: {e}")


def test_different_loading_methods():
    """
    Try different methods to load the dataset
    """
    repo_name = "BrianMwangi/kikuyu-transcription-dataset"

    print("🧪 Testing different loading methods...")

    # Method 1: Direct loading with streaming
    print("\n1️⃣ Trying streaming=True...")
    try:
        ds = load_dataset(repo_name, streaming=True)
        print("✅ Streaming load successful!")

        # Get first sample
        train_iter = iter(ds['train'])
        sample = next(train_iter)
        print(f"   Sample keys: {list(sample.keys())}")
        return ds
    except Exception as e:
        print(f"❌ Streaming failed: {e}")

    # Method 2: Force download
    print("\n2️⃣ Trying with download_mode='force_redownload'...")
    try:
        ds = load_dataset(repo_name, download_mode='force_redownload')
        print("✅ Force redownload successful!")
        return ds
    except Exception as e:
        print(f"❌ Force redownload failed: {e}")

    # Method 3: Load without audio casting first
    print("\n3️⃣ Trying to load raw data first...")
    try:
        ds = load_dataset(repo_name, cast_audio=False)
        print("✅ Raw data load successful!")
        return ds
    except Exception as e:
        print(f"❌ Raw data load failed: {e}")

    return None


def recreate_dataset_properly():
    """
    Recreate the dataset with proper structure
    """
    print("🔄 Recreating dataset with proper structure...")

    # Read local data
    excel_file = "kikuyu-dataset.xlsx"
    audio_dir = "audio"

    if not os.path.exists(excel_file):
        print(f"❌ {excel_file} not found. Make sure you're in the right directory.")
        return None

    # Read Excel file
    df = pd.read_excel(excel_file)

    # Fix audio paths if needed
    def fix_audio_path(filename):
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        return f"{name_without_ext}.wav"

    if 'wav_filename' in df.columns:
        df['audio'] = df['wav_filename'].apply(fix_audio_path)

    # Prepare dataset data
    dataset_data = []
    missing_files = []

    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['audio'])
        if os.path.exists(audio_path):
            dataset_data.append({
                'audio': audio_path,
                'transcription': row['transcription'],
                'speaker_id': str(row['speaker_id'])
            })
        else:
            missing_files.append(audio_path)

    if missing_files:
        print(f"⚠️  Missing {len(missing_files)} audio files")
        for file in missing_files[:5]:
            print(f"   - {file}")

    print(f"📊 Creating dataset with {len(dataset_data)} samples...")

    # Create dataset
    dataset = Dataset.from_list(dataset_data)

    # Cast audio column properly
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": dataset
    })

    return dataset_dict


def upload_fixed_dataset(dataset_dict, repo_name="BrianMwangi/kikuyu-transcription-dataset"):
    """
    Upload the properly structured dataset
    """
    print(f"🚀 Uploading fixed dataset to {repo_name}...")

    try:
        # Push to hub with proper configuration
        dataset_dict.push_to_hub(
            repo_name,
            private=False,
            commit_message="Fix dataset structure to resolve loading issues",
            max_shard_size="500MB"  # Ensure proper sharding
        )

        print("✅ Upload successful!")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def create_dataset_yaml(repo_name="BrianMwangi/kikuyu-transcription-dataset"):
    """
    Create a proper dataset_infos.json and README.md
    """
    print("📝 Creating dataset configuration files...")

    # Create a comprehensive README.md
    readme_content = f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- ki
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- speech
- audio
- kikuyu
- asr
size_categories:
- n<1K
---

# Kikuyu Transcription Dataset

This dataset contains Kikuyu language audio recordings with transcriptions.

## Dataset Structure

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("{repo_name}")

# Access a sample
sample = ds["train"][0]
print(sample.keys())  # ['audio', 'transcription', 'speaker_id']
```

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_name}")

# Get first sample
sample = dataset["train"][0]
audio_array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]
transcription = sample["transcription"]
speaker_id = sample["speaker_id"]
```

## License

This dataset is released under CC BY 4.0 license.
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("✅ Created README.md")


def main():
    """
    Main troubleshooting function
    """
    print("🔧 Kikuyu Dataset Loading Issue Troubleshooter")
    print("=" * 50)

    # Step 1: Clear cache and try different loading methods
    clear_cache_and_retry()

    # Step 2: Test different loading approaches
    dataset = test_different_loading_methods()

    if dataset:
        print("🎉 Dataset loaded successfully with alternative method!")
        return

    print("\n❌ All loading methods failed. Let's recreate the dataset...")

    # Step 3: Recreate dataset locally
    dataset_dict = recreate_dataset_properly()

    if not dataset_dict:
        print("❌ Could not recreate dataset. Check your local files.")
        return

    # Step 4: Create proper configuration
    create_dataset_yaml()

    # Step 5: Re-upload with fixes
    print("\n🔄 Re-uploading dataset with fixes...")

    # Ask user confirmation
    response = input("Do you want to re-upload the dataset? (y/n): ")
    if response.lower() == 'y':
        success = upload_fixed_dataset(dataset_dict)

        if success:
            print("\n✅ Dataset re-uploaded successfully!")
            print("🧪 Testing the new upload...")

            # Test the new upload
            try:
                import time
                print("⏳ Waiting 10 seconds for propagation...")
                time.sleep(10)

                test_ds = load_dataset(
                    "BrianMwangi/kikuyu-transcription-dataset", split="train[:1]")
                print("🎉 New dataset loads correctly!")

                sample = test_ds[0]
                print(f"✅ Sample loaded:")
                print(f"   - Audio shape: {sample['audio']['array'].shape}")
                print(
                    f"   - Sampling rate: {sample['audio']['sampling_rate']}")
                print(f"   - Transcription: {sample['transcription'][:50]}...")

            except Exception as e:
                print(f"⚠️  Still having issues: {e}")
                print("💡 Try waiting a few minutes and then loading with streaming=True")

# Alternative quick fix function


def quick_load_fix():
    """
    Quick fix for immediate use
    """
    print("🚀 Quick fix - Loading with streaming...")

    try:
        # Load with streaming to avoid cache issues
        ds = load_dataset(
            "BrianMwangi/kikuyu-transcription-dataset", streaming=True)

        print("✅ Streaming load successful!")

        # Convert to regular dataset if needed
        train_data = []
        for i, sample in enumerate(ds['train']):
            train_data.append(sample)
            if i >= 100:  # Limit for demo
                break

        regular_ds = Dataset.from_list(train_data)
        print(f"✅ Converted to regular dataset with {len(regular_ds)} samples")

        return regular_ds

    except Exception as e:
        print(f"❌ Quick fix failed: {e}")
        return None


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Full troubleshooting and fix")
    print("2. Quick streaming fix")

    choice = input("Enter choice (1 or 2): ")

    if choice == "2":
        dataset = quick_load_fix()
        if dataset:
            print("🎉 Quick fix successful! You can now use the dataset.")
    else:
        main()

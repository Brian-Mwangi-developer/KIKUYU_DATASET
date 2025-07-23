#!/usr/bin/env python3
"""
Simple Hugging Face Upload Script for Kikuyu Dataset
"""

import pandas as pd
import os
from datasets import Dataset, Audio, DatasetDict
from huggingface_hub import HfApi, login


def upload_kikuyu_dataset():
    """
    Upload the corrected Kikuyu dataset to Hugging Face
    """

    # Configuration
    REPO_NAME = "BrianMwangi/kikuyu-transcription-dataset"
    EXCEL_FILE = "kikuyu-dataset.xlsx"
    AUDIO_DIR = "audio"

    print(f"🚀 Uploading Kikuyu Dataset to: {REPO_NAME}")

    # Read the corrected Excel file
    print("📊 Reading dataset...")
    df = pd.read_excel(EXCEL_FILE)

    # Prepare dataset
    print("🔄 Preparing Hugging Face dataset...")
    dataset_data = []

    for _, row in df.iterrows():
        audio_path = os.path.join(AUDIO_DIR, row['audio'])
        if os.path.exists(audio_path):
            dataset_data.append({
                'audio': audio_path,
                'transcription': row['transcription'],
                'speaker_id': str(row['speaker_id'])  # Ensure it's a string
            })
        else:
            print(f"⚠️  Missing audio file: {audio_path}")

    # Create dataset
    dataset = Dataset.from_list(dataset_data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": dataset
    })

    print(f"✅ Dataset prepared with {len(dataset)} samples")

    # Upload to Hugging Face
    try:
        print("🤗 Uploading to Hugging Face Hub...")
        dataset_dict.push_to_hub(
            REPO_NAME,
            private=False,
            commit_message="Upload corrected Kikuyu transcription dataset with fixed audio paths"
        )

        print(f"🎉 Successfully uploaded!")
        print(f"📍 Dataset URL: https://huggingface.co/datasets/{REPO_NAME}")

        # Test loading
        print("\n🧪 Testing dataset loading...")
        from datasets import load_dataset
        test_ds = load_dataset(REPO_NAME, split="train[:1]")
        sample = test_ds[0]

        print("✅ Dataset loads correctly!")
        print(f"   - Audio shape: {sample['audio']['array'].shape}")
        print(f"   - Sampling rate: {sample['audio']['sampling_rate']}")
        print(f"   - Transcription: {sample['transcription'][:50]}...")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify repository permissions")


if __name__ == "__main__":
    # Make sure you're logged in first
    print("🔐 Make sure you're logged in to Hugging Face:")
    print("   Run: huggingface-cli login")
    print("   Or use: login(token='your_token')")

    input("Press Enter to continue once you're logged in...")

    upload_kikuyu_dataset()

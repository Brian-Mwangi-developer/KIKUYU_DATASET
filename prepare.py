import os
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import HfApi, login


def fix_excel_file(excel_path="kikuyu-dataset.xlsx", output_path="kikuyu-dataset.xlsx"):
    """
    Fix the Excel file to have correct audio file references
    """
    print("📊 Reading and fixing Excel file...")

    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Function to clean the filename
    def fix_audio_path(filename):
        # Extract just the filename without path and change extension to .wav
        base_name = os.path.basename(filename)  # Gets "GEN_001-1-1.mp3"
        name_without_ext = os.path.splitext(base_name)[0]  # Gets "GEN_001-1-1"
        # Return just the filename with .wav extension (no path prefix)
        return f"{name_without_ext}.wav"

    # Apply the fix to the wav_filename column
    df['wav_filename'] = df['wav_filename'].apply(fix_audio_path)

    # Rename column to 'audio' for better Hugging Face compatibility
    df = df.rename(columns={'wav_filename': 'audio'})

    # Save the corrected Excel file
    df.to_excel(output_path, index=False)
    print(f"✅ Fixed Excel file saved to: {output_path}")

    return df


def create_metadata_txt(df, output_path="metadata.txt"):
    """
    Create a pipe-delimited metadata.txt file
    """
    print("📝 Creating metadata.txt file...")

    # Create pipe-delimited format: audio|transcription|speaker_id
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            line = f"audio/{row['audio']}|{row['transcription']}|{row['speaker_id']}\n"
            f.write(line)

    print(f"✅ Metadata file created: {output_path}")


def verify_audio_files(df, audio_dir="audio"):
    """
    Verify that all audio files exist and get basic info
    """
    print("🔍 Verifying audio files...")

    missing_files = []
    total_duration = 0
    file_count = 0

    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['audio'])

        if os.path.exists(audio_path):
            try:
                # Get duration without loading the full audio
                duration = librosa.get_duration(path=audio_path)
                total_duration += duration
                file_count += 1
            except Exception as e:
                print(f"⚠️  Error reading {audio_path}: {e}")
        else:
            missing_files.append(audio_path)

    if missing_files:
        print(f"❌ Missing files ({len(missing_files)}):")
        for file in missing_files[:5]:  # Show first 5
            print(f"   - {file}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
    else:
        print("✅ All audio files found!")

    print(f"📈 Dataset stats:")
    print(f"   - Total files: {file_count}")
    print(f"   - Total duration: {total_duration/3600:.2f} hours")
    print(
        f"   - Average duration: {total_duration/file_count:.2f} seconds per file")

    return len(missing_files) == 0


def create_huggingface_dataset(df, audio_dir="audio"):
    """
    Create a Hugging Face dataset
    """
    print("🤗 Creating Hugging Face dataset...")

    # Prepare the data with full paths for audio files
    dataset_data = []
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['audio'])
        if os.path.exists(audio_path):
            dataset_data.append({
                'audio': audio_path,
                'transcription': row['transcription'],
                'speaker_id': row['speaker_id']
            })

    # Create dataset
    dataset = Dataset.from_list(dataset_data)

    # Cast audio column to Audio type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"✅ Dataset created with {len(dataset)} samples")
    return dataset


def upload_to_huggingface(dataset, repo_name, token=None):
    """
    Upload dataset to Hugging Face Hub
    """
    print(f"🚀 Uploading to Hugging Face: {repo_name}")

    # Login if token provided
    if token:
        login(token=token)
    else:
        print("💡 Make sure you're logged in: huggingface-cli login")

    try:
        # Create dataset dict (just train split for now)
        dataset_dict = DatasetDict({
            "train": dataset
        })

        # Push to hub
        dataset_dict.push_to_hub(
            repo_name,
            private=False,  # Set to True if you want a private dataset
            commit_message="Upload Kikuyu transcription dataset"
        )

        print(
            f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("Make sure you have the right permissions and are logged in.")


def create_dataset_card(repo_name="BrianMwangi/kikuyu-transcription-dataset"):
    """
    Create an updated dataset card
    """
    card_content = f"""---
language:
- ki
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- speech
- kikuyu
- asr
- transcription
size_categories:
- n<1K
---

# Kikuyu Transcription Dataset

## Dataset Summary

A collection of spoken Kikuyu utterances with transcriptions derived from Genesis biblical text. This dataset contains audio recordings in Kikuyu language paired with their corresponding transcriptions, suitable for automatic speech recognition (ASR) tasks.

## Dataset Structure

Each record contains:
- `audio`: Audio file (WAV format, 16kHz sampling rate)
- `transcription`: Kikuyu text transcription
- `speaker_id`: Speaker identifier

## Files

- `kikuyu-dataset.xlsx` – Excel spreadsheet with metadata
- `metadata.txt` – Pipe-delimited manifest file
- Audio files in WAV format (16kHz, mono)

## Usage

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_name}")

# Access the first sample
sample = dataset["train"][0]
audio = sample["audio"]
text = sample["transcription"]
speaker = sample["speaker_id"]

print(f"Transcription: {{text}}")
print(f"Speaker ID: {{speaker}}")
print(f"Audio array shape: {{audio['array'].shape}}")
print(f"Sampling rate: {{audio['sampling_rate']}}")
```

### Load with metadata.txt

```python
import pandas as pd
from datasets import load_dataset

# Load using the metadata file
ds = load_dataset("csv", 
                  data_files="{repo_name}/resolve/main/metadata.txt",
                  delimiter="|",
                  column_names=["audio", "transcription", "speaker_id"])
```

## Language

- **Language**: Kikuyu (ki)
- **Family**: Niger-Congo, Atlantic-Congo, Volta-Congo, Benue-Congo, Bantoid, Southern, Narrow Bantu, Central, J, Kikuyu-Kamba (J.20)

## Dataset Statistics

- **Total samples**: [Will be filled automatically]
- **Total duration**: [Will be calculated]
- **Speakers**: 1 primary speaker (ID: 111)
- **Domain**: Religious text (Genesis)

## Licensing and Attribution

This dataset is derived from resources provided by **OpenBible** and is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

We thank OpenBible (https://open.bible/) for their work in making biblical resources available in multiple languages.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{kikuyu_transcription_dataset,
  title={{Kikuyu Transcription Dataset}},
  author={{Brian Mwangi}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_name}}},
  license={{CC-BY-4.0}}
}}
```
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(card_content)

    print("✅ Dataset card created: README.md")


def main():
    """
    Main function to run the complete pipeline
    """
    print("🎯 Starting Kikuyu Dataset Preparation Pipeline")
    print("=" * 50)

    # Step 1: Fix Excel file
    df = fix_excel_file()

    # Step 2: Create metadata.txt
    create_metadata_txt(df)

    # Step 3: Verify audio files
    if not verify_audio_files(df):
        print("⚠️  Some audio files are missing. Please check your audio directory.")
        return

    # Step 4: Create dataset card
    create_dataset_card()

    # Step 5: Create Hugging Face dataset
    dataset = create_huggingface_dataset(df)

    # Step 6: Upload to Hugging Face (optional - uncomment and provide details)
    # REPO_NAME = "BrianMwangi/kikuyu-transcription-dataset"
    # HF_TOKEN = "your_token_here"  # Or use huggingface-cli login
    # upload_to_huggingface(dataset, REPO_NAME, HF_TOKEN)

    print("\n🎉 Pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Review the generated files (kikuyu-dataset.xlsx, metadata.txt, README.md)")
    print("2. Login to Hugging Face: huggingface-cli login")
    print("3. Uncomment the upload section in main() and run again to upload")


if __name__ == "__main__":
    main()

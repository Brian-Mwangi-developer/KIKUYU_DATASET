# Kikuyu Transcription Dataset

![](https://img.shields.io/badge/size-XXGB-blue) ![](https://img.shields.io/badge/splits-train-green)

## Dataset Summary

A collection of spoken Kikuyu utterances with transcriptions.  
Each record:  
`audio/<id>.wav|transcription|speaker_id`

## Files

- `kikuyu-dataset.xlsx` – original spreadsheet
- `metadata.txt` – pipe-delimited manifest
- `audio/` – 24 kHz, mono WAV files

## Dataset Card for HuggingFace Datasets

```yaml
name: kikuyu_transcription
version: 1.0.0
license: cc-by-4.0
task_categories: speech_recognition
languages: [ki]
homepage: https://huggingface.co/datasets/BrianMwangi/kikuyu-transcription-dataset
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("BrianMwangi/kikuyu-transcription-dataset", data_files="metadata.txt")
audio = ds["train"][0]["wav_filename"]   # path in repo
text  = ds["train"][0]["transcription"]
```

## Acknowledgments

This dataset is derived from resources provided by **OpenBible**. We thank them for their work and for making the data available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

https://open.bible/

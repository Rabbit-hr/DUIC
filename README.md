
# Partitioning by Preference: Discovering Personalized Document Clusters via User-Descriptive Clustering Intentions

## Brief introduction

This repository contains the implementation of the personalized document clustering model, **PCDI** (Partitioning by Preference: Discovering Personalized Document Clusters via User-Descriptive Clustering Intentions). PCDI is a novel approach designed to enhance user autonomy and engagement in personalized clustering. The model addresses the complex challenge of understanding and leveraging user-descriptive intentions to guide document clustering.

PCDI comprises two key components:
1. **User Intention Parser (`UIP.py`)**: Translates user-descriptive clustering intentions into actionable guidance for the clustering model.
2. **Intent-Guided Deep Semi-Supervised Clustering Module (`IGSSC.py`)**: Jointly learns document representations and cluster partitions through intent guidance.

## Project Structure

- `PCDI.py`: Main script to run the PCDI model.
- `UIP.py`: Implements the whole model: Personalized Document Clustering Model via User Descriptive Intentions
- `IGSSC.py`: Implements the Intent-Guided Deep Semi-Supervised Clustering Module.
- `run.py`: Main script to run the PCDI model.
- `requirements.txt`: List of dependencies required to run the code.

## Usage
Install the dependencies required in the requirements.txt file.

```bash
python run.py
```
This command will initiate the PCDI model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Additional Information

If you plan to run experiments with a new dataset, please ensure that you download and load the pre-trained Wikipedia Word2Vec model to the `model\English_wiki\wiki.en.vec` path. We recommend using the English Wikipedia model provided by Facebook's open-source fastText.

### Download Link

You can download the English Wikipedia model from the following link:

[fastText Wikipedia Model Download](https://fasttext.cc/docs/en/pretrained-vectors.html#wikipedia-models)

After downloading, place the model file in the `model\English_wiki\` directory and ensure the file is named `wiki.en.vec` so the program can correctly load the model.
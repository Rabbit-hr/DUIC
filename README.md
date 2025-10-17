
# DUIC: User-Descriptive Intention Guided Clustering for Personalized and Understandable Document Partitions

## Brief introduction

This repository contains the implementation of the personalized document clustering model, **DUIC** (Descriptive-Intention-Guided Understandable Clustering). DUIC is a novel approach designed to addresse two major limitations in existing personalized document clustering methods: (1) user-descriptive intention cannot be directly used to guide the clustering process, and (2) users cannot understand what each cluster represents. DUIC allows users to specify their clustering intentions through natural language descriptions and generates clusters that align with these intentions, while simultaneously providing intuitive, human-understandable explanations.

DUIC comprises two key components:
1. **Personalized Intention-Guided Clustering ('PIGC')**: Parses user intent in natural language to guide clustering that aligns with user expectations.
2. **User‑Aligned Cluster Explanation ('UACE')**: Explains clusters in a user-friendly way and provides user-aligned explanations.

## Project Structure

- `DUIC_main.py`: Main script to run the DUIC model.
- `PIGC`: Implements the Personalized Intention-Guided Clustering model.
- `PIGC/Section3_2_1_User_intent_parsing.py`: Corresponding to the content of section 3.2.1
- `PIGC/Section3_2_2_Intention_distribution_generating.py`: Corresponding to the content of section 3.2.2
- `PIGC/Section3_2_3_Distribution_Results_generating.py`: Corresponding to the content of section 3.2.3
- `UAGE`: Implements the User‑Aligned Cluster Explanation module.
- `UAGE/Section3_3_User_Aligned_Cluster_Explanation.py`: Corresponding to the content of section 3.3
- `requirements.txt`: List of dependencies required to run the code.

## Usage
Install the dependencies required in the requirements.txt file.

```bash
python DUIC_main.py
```
This command will initiate the DUIC model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Additional Information

If you plan to run experiments with a new dataset, please ensure that you download and load the pre-trained Wikipedia Word2Vec model to the `model\English_wiki\wiki.en.vec` path. We recommend using the English Wikipedia model provided by Facebook's open-source fastText.
The compressed files need to be decompressed before use. The remaining data cannot be uploaded to GitHub because it is too large. Please download it according to the link provided in the article.

### Download Link

You can download the English Wikipedia model from the following link:

[fastText Wikipedia Model Download](https://fasttext.cc/docs/en/pretrained-vectors.html#wikipedia-models)

After downloading, place the model file in the `model\English_wiki\` directory and ensure the file is named `wiki.en.vec` so the program can correctly load the model.

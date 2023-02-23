# Location Inference Transformer

## Description
**L**ocation **I**nference **T**ransformer, LIT — model that is able to determine how the word location relates to the text.

## Local usage

### Preparations
In order to use the model, you need to follow a few simple steps:

* Get vocabulary dataset for transformer.
    * Download it from [Google.Drive](https://docs.google.com/spreadsheets/d/1UBSzcdvjovl9k8AdzBHGWjf1i4WHzAue2pKR2dDClUQ) or [Kaggle](https://www.kaggle.com/viacheslavshalamov/coco-locations/).
    * Create `data` folder in root of repository and place downloaded dataset there
* Get pre-trained on [COCO Dataset](https://cocodataset.org/#home) model:
    * Download it from [Google.Drive](https://drive.google.com/file/d/17rowfXfamLQTCZO-xaBRVCaV5N_YwZ4U)
    * Create `models` folder in root of repository and place downloaded model there
* Install the appropriate version of the [PyTorch](https://pytorch.org/get-started/locally/) library for your system. 
* Install the remaining required dependencies using the command in root of repository:
    ```bash
    pip install -r requirements.txt
    ```
### Usage

#### Commands
The model is used by the command:
```bash
python launch.py -raw_text "Example text" -location "Location"
```
* `-raw_text` — required text to process
* `-location` — required location to check if it is related to text as location

Also, using an additional optional flag `-verbose`, you can enable the logging mode to get more information:
```bash
python launch.py -raw_text "Example text." -location "Location" -verbose True
```

#### Usage example:

Command:
```bash 
python launch.py -raw_text "People are skiing on the mountain." -location "mountain"
```
Output:
```
Inferening location from text: People are skiing on the mountain.
Candidate: mountain
Location: mountain is relevant location to the text
```

Command:
```bash 
python launch.py -raw_text "People are skiing on the mountain." -location "beach"
```
Output:
```
Inferening location from text: People are skiing on the mountain.
Candidate: beach
Location: mountain is not relevant location to the text
```

Command:
```bash 
python launch.py -raw_text "People are skiing on the mountain." -location "mountain" -verbose True
```
Output:
```
Inferening location from text: People are skiing on the mountain.
Candidate: mountain
Device: cuda selected
Vocabulary: COCO vocabulary loaded
Model: last epoch loaded
Text: tokenized
Location: mountain is relevant location to the text
```
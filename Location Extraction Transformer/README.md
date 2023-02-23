# Location Extraction Transformer

## Description
**L**ocation **E**xtraction **T**ransformer, LET — model capable of determining for each word in a sentence whether it can be a location.

## Local usage

### Preparations
In order to use the model, you need to follow a few simple steps:

* Get pre-trained on [COCO Dataset](https://cocodataset.org/#home) model:
    * Download it from [Google.Drive](https://drive.google.com/file/d/1qNg0um3k5SezEUU53TXxO62Q8GD9t0Ts)
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
python launch.py -raw_text "Example text"
```
* `-raw_text` — required text to process

Also, using an additional optional flag `-verbose`, you can enable the logging mode to get more information:
```bash
python launch.py -raw_text "Example text." -verbose True
```

#### Usage example:

Command:
```bash 
python launch.py -raw_text "People are skiing on the mountain."
```
Output:
```
Extracting location from text: People are skiing on the mountain.
Extracted location candidates: mountain
```

Command:
```bash 
python launch.py -raw_text "People are skiing on the mountain." -verbose True
```
Output:
```
Extracting location from text: People are skiing on the mountain.
Device: cuda selected
Model: last epoch loaded
Text: tokenized
Locations: extracted
Extracted location candidates: mountain
```
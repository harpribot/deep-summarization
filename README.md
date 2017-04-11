# Deep Summarization
Uses Recurrent Neural Network (LSTM and GRU units) for developing Seq2Seq Encoder Decoded model with and without attention mechanism for summarization of amazon food reviews into abstractive tips.

## Contents
- [Encoder Decoder Model](#encoder-decoder-model)
- [DataSet](#dataset)
- [Installation Requirements](#installation-requirements)
- [Run Instructions](#run-instructions)
- [Documentation](#documentation)
- [References](#references)

## Encoder Decoder Model
![Model](/assets/encoderdecoder.png)

## DataSet
- **DataSet Information** - This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plaintext review.

The dataset can be downloaded from [here](https://snap.stanford.edu/data/web-FineFoods.html)

A sample dataset example looks like this -
```
product/productId: B001E4KFG0
review/userId: A3SGXH7AUHU8GW
review/profileName: delmartian
review/helpfulness: 1/1
review/score: 5.0
review/time: 1303862400
review/summary: Good Quality Dog Food
review/text: I have bought several of the Vitality canned dog food products and have
found them all to be of good quality. The product looks more like a stew than a
processed meat and it smells better. My Labrador is finicky and she appreciates this
product better than most.
```

The input review has key `review/text` and the target summary that we wish to generate has key `review/summary`. For the purpose of this project, all other fields are ignored and the following two fields are extracted by the extracter script provided.


## Installation Requirements
1) Create a barebone virtual environment and activate it
```
virtualenv deepsum --no-site-packages
source deepsum/bin/activate
```

2) Install the project requirements
```
pip install -r requirements.txt
```

## Run Instructions

1) Extract the reviews and target tips using the following command
```
python extracter_script.py raw_data/finefoods.txt extracted_data/review_summary.csv
```
NOTE: Don't forget extracting the dataset and keeping it in the raw_data directory before running the above command.

2) Then run the seed script to create the required permuted training and testing dataset and also train and evaluate the model
```
# Simple - No Attention
python train_scripts/train_script_gru_simple_no_attn.py
```
This runs the Simple GRU Cell Based (Without Attention Mechanism) Encoder Decoder model.

3) Once the above script has completed execution run one of the following scripts in whichever order desired.

- For Models without Attention Mechanism

```
# Simple - No Attention
python train_scripts/train_script_lstm_simple_no_attn.py

# Stacked Simple - No Attention
python train_scripts/train_script_gru_stacked_simple_no_attn.py
python train_scripts/train_script_lstm_stacked_simple_no_attention.py

# Bidirectional - No Attention
python train_scripts/train_script_gru_bidirectional_no_attn.py
python train_scripts/train_script_lstm_bidirectional_no_attn.py

# Stacked Bidirectional - No Attention
python train_scripts/train_script_gru_stacked_bidirectional_no_attn.py
python train_scripts/train_script_lstm_stacked_bidirectional_no_attention.py

```

- For Models with Attention Mechanism

```
# Simple - Attention
python train_scripts/train_script_gru_simple_attn.py
python train_scripts/train_script_lstm_simple_attn.py

# Stacked Simple - Attention
python train_scripts/train_script_gru_stacked_simple_attn.py
python train_scripts/train_script_lstm_stacked_simple_attention.py

# Bidirectional - Attention
python train_scripts/train_script_gru_bidirectional_attn.py
python train_scripts/train_script_lstm_bidirectional_attn.py

# Stacked Bidirectional - Attention
python train_scripts/train_script_gru_stacked_bidirectional_attn.py
python train_scripts/train_script_lstm_stacked_bidirectional_attention.py
```

4) Finally exit the virtual environment once you have completed the project. You can reactivate the env later.
```
deactivate
```

## Documentation
The documentation was created automatically, and thus can be error prone. Please report any in the issue table. Some methods have missing documentation. This is not an error, but laziness on my part. I will add those documentations, when I get some free time.

To access documentation, just open index.html located at
```
docs/build/html/index.html
```
on your favorite browser. You can open them locally for now. I will try hosting them on Github pages once i get time. 
## References
1) J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.

2) Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014.

3) Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." arXiv preprint arXiv:1406.1078 (2014).

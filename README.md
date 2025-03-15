# ECGR-5106 Homework 4

## Student Information
**Name:** Yang Xu  
**Student ID:** 801443244  
**Homework Number:** 4  

## GitHub Repository
[https://github.com/yourusername/ecgr5106-hw4](https://github.com/yourusername/ecgr5106-hw4)

---

## Introduction

In this homework, explore sequence-to-sequence (seq2seq) modeling for machine translation using a GRU-based encoder-decoder architecture, both **with** and **without** attention. **All models in this homework are word-based**. Address the following problems:

- **Problem 1**: English → French translation (no attention)  
- **Problem 2**: English → French translation (with attention)  
- **Problem 3**: French → English translation (both no attention and with attention)

Use the provided dataset `Dataset-English_to_French.txt`, which contains parallel sentences in English and French. Our goals are to:

1. Train the models on the entire dataset (no train/validation split is needed given the small size).  
2. Report training loss, validation loss, and token-level validation accuracy.  
3. Generate qualitative translations to compare with the reference sentences.

In addition, based on the complete epoch logs provided, summarize the epoch at which the best accuracy was reached and compare the performance data across all 100 epochs.

---

## Problem 1: GRU-based Encoder-Decoder (English → French) - Word-Based

### 1.1 Implementation
- **Notebook**: `p1_GRU_etf.ipynb`
- **Model**: A simple GRU-based seq2seq architecture:
  - **Encoder**: Embedding layer + GRU, outputs the final hidden state.
  - **Decoder**: Embedding layer + GRU, predicts one word at a time until the EOS token is generated.
- **Vocabulary**: Constructed from **both** English and French words in the dataset by mapping each word to an index.
- **Token-Level Accuracy**: Computed as the ratio of correctly predicted words (position-wise) to the total number of target words.

### 1.2 Training Setup
- **Hidden Size**: 256  
- **Optimizer**: SGD with learning rate 0.01  
- **Loss**: NLLLoss (word-level)  
- **Epochs**: 100  

### 1.3 Results
Below is the final training vs. validation loss curve and token-level accuracy for **p1_GRU_etf**:

![Problem 1: GRU (English->French)](images/p1_GRU_etf_output.png)

**Performance Summary:**
- **Best Accuracy**: Reached at Epoch 41 (Token Accuracy = 1.0000)
  - Training loss at best epoch: ~0.0408  
  - Validation loss at best epoch: ~0.0379
- **Final Epoch (Epoch 100)**:
  - Training loss: ~0.0122  
  - Validation loss: ~0.0119  
  - Token-level validation accuracy: 1.0000

The training logs (see `output.txt`) show a steady decrease in loss and that the model converges to near-perfect accuracy on this small dataset.

### 1.4 Sample Translations
Some qualitative examples from the notebook:

``` text
Sample Translations:
Original: He checks his email
Target  : Il vérifie ses emails
Predicted: Il vérifie ses emails
--------------------------------------------------
Original: The teacher explains the lesson
Target  : Le professeur explique la leçon
Predicted: Le professeur explique la leçon
--------------------------------------------------
Original: The dog barks loudly
Target  : Le chien aboie bruyamment
Predicted: Le chien aboie bruyamment
--------------------------------------------------
Original: The baby sleeps peacefully
Target  : Le bébé dort paisiblement
Predicted: Le bébé dort paisiblement
--------------------------------------------------
Original: We plant flowers in the garden
Target  : Nous plantons des fleurs dans le jardin
Predicted: Nous plantons des fleurs dans le jardin
--------------------------------------------------
```


The predictions match the target translations exactly.

---

## Problem 2: GRU-based Encoder-Decoder with Attention (English → French) - Word-Based

### 2.1 Implementation
- **Notebook**: `p2_GRU_attention_etf.ipynb`
- **Model**: An extension of the above GRU-based model with an **Attention** mechanism:
  - The **Encoder** stores outputs at each time step.
  - The **Decoder** computes attention weights over these outputs at every decoding step, forming a context vector.
  - The context vector is concatenated with the current embedding before being input to the GRU.
  
### 2.2 Training Setup
- **Hidden Size**: 256  
- **Optimizer**: SGD (lr = 0.01)  
- **Loss**: NLLLoss  
- **Epochs**: 100  

### 2.3 Results
The following figure shows the training/validation loss and token-level accuracy for **p2_GRU_attention_etf**:

![Problem 2: GRU + Attention (English->French)](images/p2_GRU_attention_etf_output.png)

**Performance Summary:**
- **Best Accuracy**: Achieved at Epoch 41 (Token Accuracy = 1.0000)
  - Training loss at best epoch: ~0.0748  
  - Validation loss at best epoch: ~0.0619
- **Final Epoch (Epoch 100)**:
  - Training loss: ~0.0114  
  - Validation loss: ~0.0107  
  - Token-level validation accuracy: 1.0000

The attention model also converges to perfect accuracy on this dataset.

### 2.4 Sample Translations
Some sample translations from the attention model:

``` text
Sample Translations (Attention Model):
Original: He works hard every day
Target  : Il travaille dur tous les jours
Predicted: Il travaille dur tous les jours
--------------------------------------------------
Original: We are friends
Target  : Nous sommes amis
Predicted: Nous sommes amis
--------------------------------------------------
Original: She dances at the party
Target  : Elle danse à la fête
Predicted: Elle danse à la fête
--------------------------------------------------
Original: She locks the door
Target  : Elle ferme la porte à clé
Predicted: Elle ferme la porte à clé
--------------------------------------------------
Original: She catches the bus
Target  : Elle attrape le bus
Predicted: Elle attrape le bus
--------------------------------------------------
```


The attention mechanism enhances the alignment between source and target tokens.

---

## Problem 3: French → English Translation - Word-Based

For **Problem 3**, reverse the translation direction. Implement both the no-attention and the attention-based models.

### 3.1 GRU-based Encoder-Decoder (French → English) - No Attention
- **Notebook**: `p3_GRU_fte.ipynb`
- **Implementation**: Identical to Problem 1 except that the dataset pairs are reversed so that the model takes French as input and English as the target.
- **Training Setup**: Same hyperparameters (hidden size = 256, lr = 0.01, epochs = 100).

#### 3.1.1 Results
![Problem 3.1: GRU (French->English)](images/p3_GRU_fte_output.png)

**Performance Summary:**
- **Best Accuracy**: Reached at Epoch 32 (Token Accuracy = 1.0000)
- **Final Epoch (Epoch 100)**:
  - Training loss: ~0.0122  
  - Validation loss: ~0.0119  
  - Token-level validation accuracy: 1.0000

#### 3.1.2 Sample Translations

``` text
Sample Translations (French-to-English):
Input (French): Il vérifie ses emails
Target (English): He checks his email
Predicted (English): He checks his email
--------------------------------------------------
Input (French): Le professeur explique la leçon
Target (English): The teacher explains the lesson
Predicted (English): The teacher explains the lesson
--------------------------------------------------
Input (French): Le chien aboie bruyamment
Target (English): The dog barks loudly
Predicted (English): The dog barks loudly
--------------------------------------------------
Input (French): Le bébé dort paisiblement
Target (English): The baby sleeps peacefully
Predicted (English): The baby sleeps peacefully
--------------------------------------------------
Input (French): Nous plantons des fleurs dans le jardin
Target (English): We plant flowers in the garden
Predicted (English): We plant flowers in the garden
--------------------------------------------------
```

### 3.2 GRU-based Encoder-Decoder with Attention (French → English) - Word-Based
- **Notebook**: `p3_GRU_attention_fte.ipynb`
- **Implementation**: Same as Problem 2 but with reversed dataset pairs.
- **Training Setup**: Hidden size = 256, lr = 0.01, epochs = 100.

#### 3.2.1 Results
![Problem 3.2: GRU + Attention (French->English)](images/p3_GRU_attention_fte_output.png)

**Performance Summary:**
- **Best Accuracy**: Achieved at Epoch 41 (Token Accuracy = 1.0000)
- **Final Epoch (Epoch 100)**:
  - Training loss: ~0.0114  
  - Validation loss: ~0.0109  
  - Token-level validation accuracy: 1.0000

#### 3.2.2 Sample Translations

``` text
Sample Translations (French-to-English, Attention Model):
Input (French): Il travaille dur tous les jours
Target (English): He works hard every day
Predicted (English): He works hard every day
--------------------------------------------------
Input (French): Nous sommes amis
Target (English): We are friends
Predicted (English): We are friends
--------------------------------------------------
Input (French): Elle danse à la fête
Target (English): She dances at the party
Predicted (English): She dances at the party
--------------------------------------------------
Input (French): Elle ferme la porte à clé
Target (English): She locks the door
Predicted (English): She locks the door
--------------------------------------------------
Input (French): Elle attrape le bus
Target (English): She catches the bus
Predicted (English): She catches the bus
--------------------------------------------------
```

---

## Analysis
- Both translation directions (English → French and French → English) converge to near-perfect token accuracy on this small dataset.
- The best accuracies for the different models were reached at Epoch 41 (for both attention-based models) and Epoch 32 for the GRU without attention in French→English.
- The comparison of the last 100 epochs shows that the training and validation losses steadily decrease and stabilize while the token-level accuracy reaches 1.0000 in all cases.
- Overall, both attention and non-attention models achieve similar results on this dataset; however, the attention mechanism provides better alignment and interpretability, which may be more beneficial for larger corpora.

---

## Detailed Data Comparison

Based on the complete epoch logs provided in `output.txt`, the summarize the following:

- **Best Epoch (Overall Best Accuracy Achieved):**
  - **Problem 1 (GRU, Eng→Fr)**: Best at Epoch 41.
  - **Problem 2 (GRU + Attention, Eng→Fr)**: Best at Epoch 41.
  - **Problem 3 (GRU, Fr→Eng)**: Best at Epoch 32.
  - **Problem 3 (GRU + Attention, Fr→Eng)**: Best at Epoch 41.

- **Comparison of Last 100 Epochs:**
  - In all four experiments, the training and validation losses decreased steadily and stabilized, with token-level accuracy reaching 1.0000 well before the 50th epoch.
  - The final 100 epochs data confirm that the models have effectively memorized the small dataset.

---

## Conclusion

This homework demonstrates that with a small dataset, both GRU-based encoder-decoder models and their attention-based variants can achieve near-perfect token-level accuracy. **All models in this assignment are word-based**, and the experiments show that the accuracy obtained through word-based training is higher than using character-based training. 

For larger datasets, a proper train/validation split and additional regularization methods would be required to prevent overfitting and to achieve robust generalization.
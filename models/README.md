# Model Architecture and Evaluation Rationale

## Model Summary

The primary classifier is a LightGBM gradient boosted decision tree model trained to predict whether a candidate business entity pair represents the exact same real world business establishment.

The model is optimized specifically for the Macro F0.5 metric, which weights precision twice as heavily as recall.

## Model Configuration

* Model File: `models/lgbm_v4_model.txt`
* Booster Type: Gradient Boosted Decision Tree (GBDT)
* Objective: Binary classification with cross entropy loss
* Number of Iterations: 1000 with early stopping at 50 rounds
* Number of Leaves: 63
* Maximum Tree Depth: 8
* Minimum Data in Leaf: 50
* Feature Fraction: 0.80
* Bagging Fraction: 0.80
* L1 Regularization (reg alpha): 0.10
* L2 Regularization (reg lambda): 1.00
* Optimal Decision Threshold: 0.81

## Feature Schema

The classifier operates on a 20 dimensional feature vector capturing lexical, phonetic, semantic, and structural similarities:

1. `tfidf_cosine`: Character 3 to 4 n gram cosine similarity between normalized names
2. `embed_cosine`: Semantic cosine similarity between 384 dimensional dense embeddings
3. `name_fuzz_ratio`: Normalized Levenshtein ratio between business names
4. `name_partial`: Partial token ratio handling prefix and substring trade names
5. `name_token_sort`: Token sort ratio invariant to legal suffix and token permutations
6. `name_token_set`: Token set ratio isolating common intersection tokens
7. `name_jaro_winkler`: Jaro Winkler metric emphasizing prefix agreements
8. `name_jaccard_3gram`: Character tri gram Jaccard intersection over union
9. `name_token_jaccard`: Word level token Jaccard similarity
10. `name_len_ratio`: Ratio of shorter string length to longer string length
11. `name_len_diff`: Absolute difference in character lengths
12. `addr_fuzz_ratio`: Normalized Levenshtein similarity on address strings
13. `addr_token_sort`: Token sorted similarity on address components
14. `addr_token_set`: Token set intersection on address tokens
15. `addr_token_jaccard`: Address word level Jaccard similarity
16. `addr_numeric_jaccard`: Jaccard overlap on numeric building and suite tokens
17. `addr_missing`: Binary indicator flagging whether address is missing or null
18. `source_is_s3`: Source origin indicator distinguishing web directory records
19. `combined_score`: Harmonic mean balancing best name and address signals
20. `high_conf_name`: Indicator for near perfect string identity above 0.95 ratio

## Validation Results

* Validation Scheme: Group Shuffle Split on Source 1 Entity ID (80 percent train, 20 percent validation)
* Validation Entities: 441,364 Source 1 business entities
* Validation Macro F0.5: 0.84590 at threshold 0.81
* Official Public Leaderboard: 0.781662

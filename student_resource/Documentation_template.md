# ML Challenge 2026: Business Entity Resolution Solution Paper

**Team Name:** Tatwik Sai  
**Author:** Tatwik Sai  
**Submission Date:** September 2026  
**Target Metric:** Macro F0.5 Score  

---

## 1. Executive Summary

This solution presents a scalable, high precision business entity resolution engine designed to link over 12 million noisy business records across three heterogeneous multilingual sources spanning the United States, India, and France. 

Our architecture implements a dual stage framework combining dense semantic vector search via GPU accelerated FAISS with character n gram sparse TF IDF query blocking, narrowing down a 10 trillion pairwise comparison space to at most 10 candidates per entity with over 96 percent recall.

A 20 dimensional LightGBM gradient boosted classifier calibrated specifically for Macro F0.5 scoring achieves a 0.8459 validation score and a 0.7816 public leaderboard score while enforcing strict singleton defense to avoid false merge penalties.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis on the 12.5 million corporate records revealed several critical challenges:

1. Multilingual and Multi Script Variations: Source 1 canonical records are primarily Latin English, while Source 2 and Source 3 contain significant proportions of non Latin Indic scripts including Tamil, Devanagari Hindi, Marathi, and Kannada, as well as unseen French corporate registers. Standard character matching scores zero against transliterated cross script variants.
2. Missing and Corrupted Attributes: Over 30 percent of Source 2 records have completely missing address fields, requiring models to evaluate name signals independently without penalizing missing physical location data.
3. Asymmetric Metric Penalty: Macro F0.5 penalizes false merges (matching two different businesses) twice as heavily as missed matches. In addition, singletons (entities with zero matches) award a full 1.0 score when left empty, but collapse to 0.0 upon any false positive prediction.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi Stage Blocking with GBDT Classification and Calibrated Singleton Defense.

**Core Innovation:**
1. Dual Dense and Sparse Inverted Indexing: Using multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) in conjunction with character 3 to 4 n gram TF IDF inverted indices to resolve both typographical noise and cross script transliterations.
2. Query Centric S1 Blocking: Treating Source 1 as query rows and indexing Source 2 and Source 3 in streaming chunks to maintain constant memory consumption below 2 GB on 10 million candidate targets.
3. Singleton Protection Guard: Absolute preservation of true singletons based on model confidence margins, preventing catastrophic precision degradation.

---

## 3. Candidate Generation (Blocking)

The candidate generation stage reduces the pairwise search space from $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.72 \times 10^{13}$ pairs to under 10 candidates per Source 1 entity (reduction ratio exceeding 99.9999 percent).

* **Blocking Channels:**
  1. Semantic Vector Search: FAISS flat inner product search over normalized 384 dimensional dense embeddings, retrieving top 10 nearest neighbors with similarity threshold 0.25.
  2. Sparse Lexical Blocking: Character 3 to 4 n gram TF IDF vectorization over normalized names with sublinear term frequency, querying Source 2 and 3 chunks via sparse matrix multiplication.
  3. Exact Inverted Name Index: Fast hash map inverted index on normalized business names to guarantee instant capture of identical corporate titles.
* **Country Partitioning:** Partitioning candidate generation by country prevents impossible international cross merges while seamlessly handling the unseen France partition.
* **Recall Retention:** Taking the union of top candidates across dense, sparse, and exact channels ensures high coverage of both lexical variants and semantic trade name aliases.

---

## 4. Matching Model

### 4.1 Feature Engineering

Candidate pairs are mapped into a 20 dimensional dense feature vector across four domains:

* **Lexical and Phonetic Similarities:**
  * Normalized Levenshtein ratio (`name_fuzz_ratio`)
  * Partial substring token ratio (`name_partial`)
  * Token sorted ratio and token set intersection ratio (`name_token_sort`, `name_token_set`)
  * Jaro Winkler similarity emphasizing prefix agreement (`name_jaro_winkler`)
  * Character tri gram Jaccard similarity (`name_jaccard_3gram`)
  * Word level token Jaccard similarity (`name_token_jaccard`)
  * Relative length ratio and absolute length difference (`name_len_ratio`, `name_len_diff`)
* **Address and Location Alignment:**
  * Address token sort and token set ratios (`addr_token_sort`, `addr_token_set`)
  * Address word level Jaccard similarity (`addr_token_jaccard`)
  * Numeric building number and suite Jaccard overlap (`addr_numeric_jaccard`)
  * Missing address indicator (`addr_missing`)
* **Semantic Representations:**
  * Multilingual sentence transformer cosine similarity (`embed_cosine`)
  * Character n gram TF IDF cosine similarity (`tfidf_cosine`)
* **Structural and Combined Features:**
  * Web directory record indicator (`source_is_s3`)
  * Harmonic mean interaction between top name and address scores (`combined_score`)
  * High confidence name indicator (`high_conf_name`)

### 4.2 Model Architecture and Threshold Calibration

* **Architecture:** LightGBM Gradient Boosted Decision Tree with 63 leaves, maximum depth 8, and L1 and L2 regularization to prevent overfitting on rare n grams.
* **Threshold Selection:** A vectorized grid sweep across validation entities identifies the global decision threshold (0.81) that maximizes Macro F0.5. Secondary matches are subjected to strict confidence criteria and tail truncation at 5 matches per entity to prevent precision decay.

---

## 5. Results and Error Analysis

* **Validation Performance:** Macro F0.5 score of 0.84590 at threshold 0.81 on 441,364 held out Source 1 validation entities.
* **Public Leaderboard:** Macro F0.5 score of 0.781662, ranking in the top tier of competitive participants.
* **False Positive Analysis:** The primary source of false merges stems from multi tenant commercial buildings sharing identical numeric street addresses where business names have ambiguous generic words (such as holding, services, or capital). Enforcing token overlap guards on non numeric tokens mitigated these errors.
* **False Negative Analysis:** Missed matches primarily involve trade name aliases with completely dissimilar branding from registered legal names that lack address data in Source 2.

---

## 6. Conclusion

This project demonstrates that combining multilingual representation learning with high throughput sparse blocking provides an effective, scalable foundation for massive business entity resolution. By prioritizing precision through calibrated thresholding and singleton defense, the pipeline satisfies stringent real world enterprise requirements and achieves superior performance under the Macro F0.5 objective.

---

## Appendix: Code Artifacts and Reproducibility

The project source code is organized modularly under `src/`:
* `src/preprocessing.py`: Multilingual normalization and address standardization
* `src/blocking.py`: Dual dense FAISS GPU and sparse TF IDF candidate blocking
* `src/features.py`: Parallel 20 dimensional feature engineering
* `src/model.py`: LightGBM training and threshold calibration
* `src/inference.py`: Test inference, candidate pairs generation, and format validation

To reproduce the submission files from training and test datasets:
```bash
python scripts/run_training.py
python scripts/run_inference.py
python scripts/validate.py submissions/sub5/matching_results.tsv submissions/sub5/candidate_pairs.tsv
```

### OVERVIEW

dataset: https://www.kaggle.com/datasets/najzeko/steam-reviews-2021

problem: Among Steam reviews whose text is negative in sentiment, what linguistic patterns distinguish reviews that recommend the game from those that do not recommend it?

hypothesis: Within negative-sentiment reviews, those that still recommend the game focus more on core gameplay quality and fun, whereas negative reviews that do not recommend focus more on bugs, performance problems, and monetization/price complaints.

data cleaning

- only take reviews in English and game_id = 730 (csgo)
- balance the number of recommended vs not recommended reviews
- remove really short reviews
- use a prebuilt sentiment model or hand label reviews to train my own classifier, then classify the reviews and only keep the negative ones

data processing

- vectorize reviews based on tf-idf or n-gram

---

### PROMPT

You are an expert data scientist and ML engineer working in a Jupyter/Python environment with access to the Kaggle dataset:

"Steam Reviews 2021" → https://www.kaggle.com/datasets/najzeko/steam-reviews-2021

Your task is to implement a complete, reproducible pipeline (ideally as a single, well-structured Jupyter notebook or a main .py file plus helpers) to test the following hypothesis (H1) using this dataset.

---

## PROJECT OVERVIEW

Research question:
Among Steam reviews whose text is negative in sentiment, what linguistic patterns distinguish reviews that recommend the game from those that do not recommend it?

Hypothesis H1 (content-focused):
Within negative-sentiment reviews, those that still recommend the game focus more on core gameplay quality and fun, whereas negative reviews that do not recommend the game focus more on bugs/performance problems and monetization/price complaints.

We operationalize this as:
• Outcome Y: whether the review recommends the game (steam’s "voted_up"/"recommended" flag).
• Condition: restrict to reviews whose text is negative according to a sentiment model.
• Predictors: complaint-type features derived from the text (lexicons and/or topic proportions), plus simple controls (e.g., review length, game fixed effects).

You should produce:

1. Data loading / sampling code.
2. Sentiment analysis and definition of the "negative subset".
3. Text feature engineering (lexicon features + topic model).
4. Descriptive comparisons between "negative+recommend" vs "negative+not recommend".
5. A logistic regression model testing H1.
6. Basic robustness knobs (e.g., adjustable sentiment threshold).

Write clean, modular, well-commented code that a student can read and modify.

---

## DATA HANDLING

1. Loading and sampling
   • Assume the raw CSV is at a configurable path, e.g. DATA_PATH = "steam_reviews.csv".
   • Use pandas to load the file, but because the dataset is large (GB-scale), implement one of:
   – A random row sample (e.g., 500k–1M reviews) using `skiprows` / chunking, or
   – Sampling per game (e.g., up to N reviews per game for the most-reviewed games).
   • Expose sample size and sampling strategy as parameters at the top of the notebook/script.
   • Keep at least the following columns (names may differ; infer from the header):
   – review text (likely "review")
   – "voted_up" (or "recommended") → boolean recommendation flag
   – "language"
   – game/app id (e.g., "appid")
   – any other useful metadata (e.g., "timestamp_created", "steam_purchase", "received_for_free", etc., if available).

2. Basic filtering
   • Filter to English reviews: `language == 'english'` (or whatever exact label is in the CSV).
   • Drop null/empty review texts.
   • Drop very short reviews (e.g., fewer than 10 tokens).

3. Provide a function:
   `load_and_sample_data(path: str, sample_size: int, random_state: int) -> pd.DataFrame`
   that returns a cleaned, sampled DataFrame ready for NLP.

---

## SENTIMENT ANALYSIS AND NEGATIVE SUBSET

Goal: assign each review a sentiment score and define a “negative review” subset.

1. Sentiment model
   • Use VADER (from `nltk.sentiment.vader`) or another standard sentiment tool available in Python.
   • Compute VADER's `compound` score in [-1, 1] for each review text.
   • Add a column `sentiment_compound`.

2. Define negativity
   • Define a negativity threshold parameter, e.g.:
   NEG_THRESHOLD = 0.0
   • Create a binary indicator:
   negative = (sentiment_compound < NEG_THRESHOLD)
   • Filter the DataFrame to only rows where `negative` is True.
   • In the code, make NEG_THRESHOLD easy to change (constant at top or function argument).

3. Group labels for H1
   • Inside the negative subset, define:
   – Group A (love–hate): negative sentiment AND `voted_up == True`
   – Group B (full rejection): negative sentiment AND `voted_up == False`
   • Add a binary column `recommended_flag` = 1 if voted_up is true, else 0.
   • Ensure you print basic counts:
   – Number of negative reviews
   – Counts of negative+recommended vs negative+not-recommended

4. Encapsulate this logic in a function:
   `add_sentiment_and_filter_negative(df: pd.DataFrame, neg_threshold: float) -> pd.DataFrame`

---

## TEXT PREPROCESSING AND FEATURE ENGINEERING

We need features capturing complaint types: bugs/performance, monetization/price, and gameplay/fun/content.

1. Text preprocessing
   • Create a text-cleaning function that:
   – lowercases
   – removes URLs
   – optionally removes punctuation except where needed for exclamation count
   – tokenizes (use a simple tokenizer, it doesn’t need to be perfect)
   • Store a cleaned text column (`clean_text`) for feature extraction.

2. Lexicon-based features (for interpretability)
   • In code, define three small lexicons (word lists) for:
   – BUG/PERFORMANCE: ["crash", "crashes", "lag", "laggy", "fps", "bug", "bugs", "glitch", "glitches", "freeze", "freezing", "stutter", "stuttering", "performance", "optimization", "optimized", ...]
   – MONETIZATION/PRICE: ["microtransaction", "microtransactions", "mtx", "lootbox", "lootboxes", "dlc", "season pass", "pay2win", "pay-to-win", "cash grab", "overpriced", "refund", "sale", "expensive", ...]
   – GAMEPLAY/FUN/CONTENT: ["fun", "gameplay", "combat", "balance", "content", "story", "graphics", "soundtrack", "music", "replay", "grind", "coop", "co-op", "friends", ...]
   • Implement a function that:
   – Tokenizes each cleaned review,
   – Counts occurrences from each lexicon,
   – Normalizes per 100 tokens (e.g., (#lexicon_words / #tokens) \* 100).
   • Add numerical columns:
   – bug_lex_per_100
   – monet_lex_per_100
   – gameplay_lex_per_100
   • Additionally, compute simple stylistic features:
   – review_len_tokens
   – exclamation_count
   – maybe profanity_count using a small predefined list of swear words.

3. Topic modeling (optional but preferred)
   • Use scikit-learn’s `CountVectorizer` + `LatentDirichletAllocation` OR BERTopic to fit a topic model on the negative subset.
   • Choose a reasonable K (e.g., 10–20 topics, set as a parameter).
   • After fitting:
   – Print top words per topic to inspect.
   – Produce a table or print-out with topic index and top 10–15 words.
   • For each review, compute topic distribution (topic proportions) and add them as features:
   – topic*0, topic_1, ..., topic*{K-1}
   • These can be used in the regression and for descriptive comparisons.

4. Structure this as functions:
   • `build_lexicon_features(df: pd.DataFrame, lexicons: dict) -> pd.DataFrame`
   • `fit_topic_model(clean_text_series: pd.Series, n_topics: int) -> (LDA_model, vectorizer, topic_proportions_df)`
   • Make sure topic_proportions_df aligns with rows of the filtered DataFrame.

---

## DESCRIPTIVE ANALYSIS FOR H1

Within the _negative_ subset:

1. Basic summaries
   • Print counts and proportions:
   – negative & recommended_flag == 1
   – negative & recommended_flag == 0

2. Lexicon features by group
   • Compute mean and standard deviation of:
   – bug_lex_per_100
   – monet_lex_per_100
   – gameplay_lex_per_100
   separately for recommended_flag == 1 vs 0.
   • Perform significance tests (e.g., scipy.stats t-test or Mann–Whitney U) on differences in means for each feature.
   • Create a small summary table of these statistics and p-values.

3. If topic modeling is used:
   • Compute average topic proportions per group (recommended_flag == 1 vs 0).
   • Visualize selected topics with barplots of mean topic proportion by group.

4. Plotting
   • Use matplotlib or seaborn to:
   – Plot distributions (e.g., boxplots or violin plots) of bug_lex_per_100, monet_lex_per_100, gameplay_lex_per_100 by recommended_flag.
   – Optionally, bar charts for topic mean differences.

---

## LOGISTIC REGRESSION FOR H1

We want a multivariate model where the outcome is recommended_flag inside the negative subset, predicted by complaint-type features.

1. Model specification
   • Use statsmodels (preferred) or scikit-learn; if possible, use statsmodels for interpretable coefficients and p-values.
   • Outcome:
   – Y_i = recommended_flag (1 = recommended, 0 = not recommended)
   • Predictors:
   – bug_lex_per_100
   – monet_lex_per_100
   – gameplay_lex_per_100
   – review_len_tokens
   – (optionally) selected topic proportions (e.g., 3–5 topics that clearly correspond to gameplay/bugs/price)
   – (optionally) game fixed effects (dummy variables for top N games, or game ID as a random effect if you prefer).

2. Fit the model
   • Standardize/scale predictors if needed.
   • Fit a logistic regression:
   – For statsmodels: use `Logit` or `GLM(family=Binomial())`.
   – Cluster standard errors by game/app id if straightforward, or at least report robust standard errors.
   • Print the full model summary (coefficients, standard errors, z-statistics, p-values).

3. Hypothesis tests for H1
   • Focus on signs and significance of complaint-type predictors:
   – Expect bug_lex_per_100 and monet_lex_per_100 to have **negative** coefficients (more bug/price complaints → lower probability of recommendation among negative reviews).
   – Expect gameplay_lex_per_100 to have a **positive** coefficient (more gameplay/fun talk → higher probability of recommendation among negative reviews).
   • Optionally, compute marginal effects or predicted probabilities for:
   – Low vs high gameplay_lex_per_100
   – Low vs high bug_lex_per_100 / monet_lex_per_100

4. Minimal predictive evaluation (optional)
   • Split the negative subset into train/test, fit the logistic model (or an equivalent sklearn model) and report:
   – Accuracy and ROC–AUC on the test set.
   • This is secondary; emphasis is on inference and coefficient interpretation, not maximizing predictive performance.

---

## ROBUSTNESS KNOBS

Implement at least the following as easy-to-change parameters:

• NEG_THRESHOLD: sentiment threshold for defining "negative" (e.g., 0.0, -0.1, -0.2).
• N_TOPICS: number of topics in the topic model.
• SAMPLE_SIZE: number of reviews to sample.

Optionally, you can:
• Re-run the logistic regression under a more conservative negativity threshold (e.g., NEG_THRESHOLD = -0.2) and show that coefficients maintain their signs and remain reasonably similar in magnitude.

---

## CODE QUALITY AND ORGANIZATION

Please:
• Structure code into clear sections with markdown headers if in a notebook:
– 0. Imports and constants
– 1. Load & sample data
– 2. Sentiment & negative subset
– 3. Text preprocessing & features
– 4. Descriptive analysis
– 5. Logistic regression for H1
– 6. (Optional) Robustness checks
• Write small helper functions instead of huge monolithic blocks.
• Add concise inline comments explaining:
– Why each step is done,
– How each feature relates back to H1.
• At the end, print a short textual summary (e.g., a few lines) interpreting the key coefficients and whether H1 is supported.

Implement this entire pipeline now: create the necessary Python code (and markdown if using a notebook) so that, once the CSV path is set and nltk’s VADER is installed, the analysis for H1 can be run end-to-end.

# Methods Section

## 3 Methods

To test the hypothesis that negative reviews recommending Rainbow Six Siege emphasize gameplay quality while negative non-recommending reviews focus on bugs, performance, and monetization, I implemented a multi-stage analytical pipeline combining sentiment analysis, lexicon-based feature extraction, topic modeling, and statistical modeling.

### 3.1 Data Preprocessing and Sentiment Filtering

The analysis began with filtering the Steam Reviews 2021 dataset to Rainbow Six Siege (app ID: 359550), English-language reviews from users who purchased the game on Steam. I removed reviews with null or empty text and those with fewer than 10 tokens to ensure sufficient content for analysis. This yielded 108,264 reviews.

To identify negative reviews, I applied VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based sentiment analyzer from the Natural Language Toolkit (Hutto & Gilbert, 2014). VADER computes a compound sentiment score ranging from -1 (most negative) to +1 (most positive). I classified reviews with compound scores less than 0.0 as negative, resulting in 24,284 negative reviews. Within this negative subset, 13,817 reviews recommended the game while 10,467 did not. To ensure balanced comparison, I randomly sampled 10,467 reviews from each group, producing a final balanced dataset of 20,934 negative reviews.

### 3.2 Feature Engineering

I extracted three types of features to capture the linguistic patterns hypothesized to distinguish the two groups.

**Lexicon-based features.** I constructed three custom lexicons corresponding to the complaint types in the hypothesis: (1) bugs and performance issues (25 terms: e.g., "crash", "lag", "fps", "bug", "stutter", "optimization"), (2) monetization and pricing concerns (24 terms: e.g., "microtransaction", "dlc", "pay2win", "overpriced", "cash grab"), and (3) gameplay quality and enjoyment (27 terms: e.g., "fun", "gameplay", "combat", "balance", "graphics", "competitive"). For each review, I counted occurrences of lexicon terms and normalized these counts per 100 tokens to control for review length, creating three features: `bug_lex_per_100`, `monet_lex_per_100`, and `gameplay_lex_per_100`.

**Stylistic features.** I computed three additional features to control for review characteristics: total token count (`review_len_tokens`), number of exclamation marks (`exclamation_count`), and profanity count using a small predefined list of common swear words (`profanity_count`).

**Topic modeling features.** To capture broader thematic patterns beyond individual words, I fit a Latent Dirichlet Allocation (LDA) topic model (Blei et al., 2003) on the negative review corpus. I first vectorized the cleaned review texts using a count vectorizer with minimum document frequency of 5, maximum document frequency of 0.8, and a vocabulary limit of 5,000 terms. I then fit an LDA model with 15 topics using batch learning. For each review, I extracted the topic proportion distribution, resulting in 15 additional features (`topic_0` through `topic_14`) representing the relative emphasis on each discovered topic.

### 3.3 Descriptive Analysis

I compared the two groups (negative+recommended vs. negative+not-recommended) on the lexicon features using descriptive statistics (means and standard deviations) and statistical tests. For each lexicon feature, I performed independent samples t-tests (or Mann-Whitney U tests if normality assumptions were violated) to assess whether group differences were statistically significant. I also examined average topic proportions by group to identify topics that distinguish the two review types.

### 3.4 Hypothesis Testing via Logistic Regression

To test the hypothesis while controlling for multiple factors simultaneously, I fit a logistic regression model predicting the recommendation status (recommended = 1, not recommended = 0) within the negative review subset. The model included as predictors the three lexicon features (`bug_lex_per_100`, `monet_lex_per_100`, `gameplay_lex_per_100`), review length as a control variable, and selected topic proportions that clearly corresponded to bugs, monetization, or gameplay themes.

The hypothesis predicts that:

- `gameplay_lex_per_100` will have a **positive** coefficient (more gameplay talk → higher probability of recommendation among negative reviews)
- `bug_lex_per_100` and `monet_lex_per_100` will have **negative** coefficients (more bug/price complaints → lower probability of recommendation among negative reviews)

I used statsmodels to fit the model and obtain interpretable coefficients with standard errors and p-values. I also computed marginal effects to quantify the practical significance of differences in lexicon usage.

### 3.5 Robustness Checks

To assess the sensitivity of results to parameter choices, I re-ran the analysis with alternative sentiment thresholds (e.g., -0.1, -0.2 for more conservative negativity definitions) and different numbers of topics (10, 20) in the LDA model, comparing whether coefficient signs and significance levels remained consistent across specifications.

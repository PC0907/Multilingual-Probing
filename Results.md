# Initial Probing Results: Simple Explanation

**Status: 7 September 2026**

## The main idea

We tested whether **Llama 3.1 8B internally represents whether statements are true or false**.

We tested English and German statements and looked at different layers of the model to see where this information is easiest to detect.

The main result is that **the pipeline works**, but the results also show that **dataset construction and topic matter a lot**.

---

## How the experiment works

The model was not trained or fine-tuned.

Instead:

1. A statement is given to the model.
2. We collect its internal activations from every layer.
3. We create a simple direction representing the difference between true and false statements.
4. We test whether that direction can correctly classify unseen statements.

We used five-fold cross-validation and kept very similar statements together in the same fold. This prevents the model from getting an unfair advantage by seeing almost identical statements during training and testing.

---

## The three main datasets

We tested:

- **English Cities**: structured geography facts.
- **English CommonClaim**: mixed, crowd-sourced claims.
- **German**: a dataset created for this project, covering geography, sports, entertainment, and history/news.

The English CommonClaim dataset was made roughly the same size as the German dataset so the comparison is fairer.

---

## Main results

| Dataset | Best accuracy | Simple word-based baseline |
|---|---:|---:|
| English Cities | 0.976 | 0.475 |
| English CommonClaim | 0.757 | 0.596 |
| German | 0.656 | 0.449 |

At first glance, German looks worse because its accuracy is lower.

However, the word-based baseline tells a more interesting story.

### English CommonClaim

A simple classifier using only the words in the statements already achieves **0.596** accuracy.

The representation probe increases this to **0.757**, which is an improvement of about **16 percentage points**.

### German

The word-based classifier gets only **0.449**, below chance.

The representation probe increases this to **0.656**, an improvement of about **21 percentage points**.

### What this means

Although German has lower raw accuracy, the probe is finding **more information beyond simple surface wording**.

So raw accuracy alone can be misleading.

---

# The pipeline is working

The English Cities dataset reaches **0.976 accuracy at layer 16**, exactly halfway through the model.

Its performance follows the expected pattern:

- Near chance at the earliest layers.
- Strong improvement through the middle layers.
- About 0.90 accuracy by layer 8.
- A peak around the middle.
- A plateau afterwards.

Because this dataset produces the expected result using the same code as the German experiment, the German result is unlikely to be caused by a bug in the pipeline.

---

# Dataset construction matters a lot

One of the strongest findings is the difference between the two English datasets:

- English Cities: **0.976**
- English CommonClaim: **0.757**

That is a difference of about **22 percentage points**.

Both experiments use:

- The same language.
- The same model.
- The same code.
- The same pooling method.

The major difference is the **way the datasets were constructed**.

This suggests that dataset construction can strongly affect truth-probing results.

Therefore, comparing two languages using differently constructed datasets can be dangerous. A difference that looks like a language effect might actually be caused by differences in the datasets.

---

# German results also depend strongly on topic

Within German, the results differ by topic:

| Topic | Accuracy |
|---|---:|
| History/News | 0.690 |
| Geography | 0.670 |
| Sports | 0.627 |
| Entertainment | 0.506 |

Entertainment is essentially at chance.

This is interesting because all four topics were constructed in a similar way and have similar numbers of examples.

The entertainment statements involve things such as German directors, comic prizes, and television channels. These facts may be more locally specific and less common in the model's training data.

However, there are currently two possible explanations:

1. **The model does not reliably know these entertainment facts.**
2. **The model knows them, but the current probe cannot read the information properly.**

A behavioural test is needed to distinguish between these possibilities.

---

# The controls look good

The shuffled-label experiments are close to chance for German and CommonClaim.

This is important because it suggests that the evaluation procedure is not leaking answers from training to testing.

Also, using grouped splits instead of random splits does not artificially improve the results.

The lexical baseline for German is below chance, which also suggests that simple wording does not provide an easy shortcut for solving the task.

---

# Current limitations

## Only one model

So far, these results are only for **Llama 3.1 8B**.

Other models are still being tested.

## Only one pooling method

We use the activation of the final token as the representation of the statement.

This may be a problem for cross-lingual comparisons because the final token can represent different linguistic positions in different languages.

## German lacks enough negated statements

The German dataset mostly contains affirmative statements.

Because of this, we cannot separate:

- Truth
- Negation/polarity

This means the planned two-dimensional truth representation and principal-angle analysis cannot currently be performed for German.

A rebuilt German dataset with both truth and negation conditions would solve this problem.

## Some shuffled results for Cities vary more than expected

The shuffled control for the Cities dataset has some unusually high and low values at individual layers.

This may simply be random noise, but it needs to be checked using several random seeds.

---

# What happens next?

The next steps are:

1. **Run a behavioural test for the German topics.**  
   Ask the model directly whether the statements are true or false. This will show whether the entertainment problem comes from missing knowledge or from the representation/probe.

2. **Repeat the experiments with multiple random seeds.**  
   This will confirm that the shuffled controls behave as expected.

3. **Decide whether to rebuild the German dataset.**  
   A new truth-by-polarity design with entity spans would allow the full cross-lingual geometry analysis.

4. **Run topic-transfer experiments within German.**  
   This will test whether a truth direction learned from one topic transfers to another topic.

---

# Bottom line

The experiments successfully validate the technical pipeline.

The most important scientific lesson so far is that **truth-probing results are strongly affected by how the dataset is constructed and what type of knowledge it contains**.

This means raw cross-lingual accuracy comparisons should be treated carefully. Before claiming that two languages represent truth differently, we need to make sure that differences are not caused by dataset construction, lexical shortcuts, or differences in knowledge domains.

The German entertainment result is currently the most interesting open question, and the behavioural check is the next experiment most likely to explain it.

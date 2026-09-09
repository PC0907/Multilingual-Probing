# How to run your model

The examples below use `qwen3-8b` as the model. Replace it with yours in every
command.

You need a GPU with at least 20 GB for steps 2 and 3. Steps 4 and 5 run on any
machine.

---

## The three datasets

All three are already in the repo. Use them as they are, do not regenerate them.

| File | What it is | Size |
| --- | --- | --- |
| `data/processed/en_cities/index.csv` | English. "The city of X is in Y", generated from a template. From the geometry-of-truth repo (Marks & Tegmark). | 1496 |
| `data/processed/en/index.csv` | English. CommonClaim, crowd-sourced claims about anything. From the Truth_is_Universal repo (Bürger et al.). | 1996 |
| `data/processed/de_v2/index.csv` | German. Written by us. Four topics: geography, sports, entertainment, history/news, 500 each. | 1996 |

Each has a `text` column and a `label` column, where **1 means true**.

---

## Step 1. Get the code

```
git clone --branch Ali https://github.com/PC0907/Multilingual-Probing.git
cd Multilingual-Probing
pip install -r requirements.txt
```

## Step 2. Check your setup works

Run the English `cities` dataset first. Other papers have published results on
it, so it tells you whether your setup is correct.

```
python scripts/02_extract.py \
  --model Qwen/Qwen3-8B \
  --index data/processed/en_cities/index.csv \
  --out data/processed/en_cities/qwen3-8b \
  --pooling last --batch-size 16

python scripts/06_layer_sweep.py \
  --cache data/processed/en_cities/qwen3-8b \
  --index data/processed/en_cities/index.csv \
  --out results/en_cities_sweep_qwen3-8b --stride 4
```

The first command writes 33 files (`layer_00.npy` to `layer_32.npy`, plus
`meta.json`) into `data/processed/en_cities/qwen3-8b/`. About 1 GB.

**You should see peak accuracy around 0.95 or higher, in the middle layers.**

If you see around 0.65, stop and tell us. Do not run the rest.

## Step 3. Run German and CommonClaim

```
python scripts/02_extract.py \
  --model Qwen/Qwen3-8B \
  --index data/processed/de_v2/index.csv \
  --out data/processed/de_v2/qwen3-8b \
  --pooling last --batch-size 16

python scripts/02_extract.py \
  --model Qwen/Qwen3-8B \
  --index data/processed/en/index.csv \
  --out data/processed/en/qwen3-8b \
  --pooling last --batch-size 16
```

## Step 4. Measure accuracy

```
python scripts/06_layer_sweep.py \
  --cache data/processed/de_v2/qwen3-8b \
  --index data/processed/de_v2/index.csv \
  --out results/de_sweep_qwen3-8b --stride 4

python scripts/06_layer_sweep.py \
  --cache data/processed/en/qwen3-8b \
  --index data/processed/en/index.csv \
  --out results/en_sweep_qwen3-8b --stride 4
```

Each writes a `.csv` and a `.json` into `results/`.

Check the `shuffled` column. It should be near 0.50 at every layer. If it is
much higher, tell us before continuing.

The German output also has one column per topic. The entertainment column is the
one we are most interested in.

## Step 5. Measure alignment

Three runs. No GPU needed, a few minutes each.

```
python scripts/07_alignment.py \
  --cache-a data/processed/de_v2/qwen3-8b \
  --index-a data/processed/de_v2/index.csv --name-a de \
  --cache-b data/processed/en/qwen3-8b \
  --index-b data/processed/en/index.csv --name-b en \
  --out results/de_en_qwen3-8b --stride 4

python scripts/07_alignment.py \
  --cache-a data/processed/de_v2/qwen3-8b \
  --index-a data/processed/de_v2/index.csv --name-a de \
  --cache-b data/processed/en_cities/qwen3-8b \
  --index-b data/processed/en_cities/index.csv --name-b cities \
  --out results/de_cities_qwen3-8b --stride 4

python scripts/07_alignment.py \
  --cache-a data/processed/en/qwen3-8b \
  --index-a data/processed/en/index.csv --name-a commonclaim \
  --cache-b data/processed/en_cities/qwen3-8b \
  --index-b data/processed/en_cities/index.csv --name-b cities \
  --out results/en_en_qwen3-8b --stride 4
```

## Step 6. Send us the results

```
tar czf results_qwen3-8b.tar.gz \
  results/ \
  data/processed/de_v2/qwen3-8b/meta.json \
  data/processed/en/qwen3-8b/meta.json \
  data/processed/en_cities/qwen3-8b/meta.json
```

Send that file. It will be small. Keep the `.npy` files on your own machine.

---

## Numbers from Llama 3.1 8B, for comparison

Yours will be different. That is expected. These are here so you can tell "my
model is different" apart from "my setup is broken".

| Step | Measurement | Result |
| --- | --- | --- |
| 2 | `en_cities` accuracy | 0.976 at layer 16 of 32 |
| 4 | `de_v2` accuracy | 0.664 at layer 12 |
| 4 | `en` accuracy | 0.757 at layer 12 |
| 4 | `de_v2` entertainment topic | 0.539, much lower than the other three |
| 5 | German vs English alignment | 0.600 at layer 16 |
| 5 | German vs cities alignment | 0.488 at layer 12 |

## Two rules

Always use `--pooling last`. The other options need information these datasets
do not carry.

Use `scripts/06_layer_sweep.py` and `scripts/07_alignment.py` rather than your
own code. If you use logistic regression or a different train/test split the
numbers will not be comparable to anyone else's.

## Known issues

In step 5, layer 0 is skipped with a message about identical class means. That is
correct, not an error. Every German statement ends in a full stop, so at the
embedding layer there is nothing to separate.

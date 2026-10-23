# Kumo Relational on SQLite: multi-table predictions without training

A notebook tutorial for **NVIDIA Kumo Relational**, a pre-trained relational foundation model.
It predicts straight from connected tables (customers → orders → items → sellers ...) by
**in-context learning**. There is no training run, no feature engineering, and no flattening into
one wide table.

The tutorial uses a real multi-table dataset, **Olist Brazilian E-Commerce** from Kaggle. The data
is loaded into a **SQLite** database with real primary and foreign keys. Then five business
questions are answered with **Predictive Query Language (PQL)**, and each one is backtested against
a classic "SQL features + gradient boosting" pipeline.

| File | What it is |
|---|---|
| [`kumo_relational_sqlite_tutorial.ipynb`](kumo_relational_sqlite_tutorial.ipynb) | The hands-on notebook (18 sections) |
| `tutorial_source.py` | Plain-text source of the notebook. Edit this, then run `python build_notebook.py` |
| `build_notebook.py` | Turns `tutorial_source.py` into the `.ipynb` |
| `README.md` | This guide: the concepts behind each notebook section |

---

## 1. Quick start

```bash
cd kumo-relational-sqlite
python -m venv .venv && source .venv/bin/activate
pip install "kumo-relational-client[relational,sqlite]" scikit-learn matplotlib kagglehub jupyter

# Real data: a Kaggle API token in ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY)
# Prediction: a Kumo Relational NIM on a GPU box
docker login nvcr.io        # user: $oauthtoken, password: <NGC API key>
docker run -it --rm --gpus all --shm-size=16GB -p 8000:8000 nvcr.io/nim/nvidia/kumo-relational:1.0.0

jupyter lab kumo_relational_sqlite_tutorial.ipynb
```

The notebook adapts to what you have:

| You have | What runs |
|---|---|
| Nothing (no Kaggle, no GPU) | A **synthetic dataset with Olist's schema**, SQLite build, SQL exploration, graph construction, offline PQL checks, and all classic baselines. Kumo cells print `⏭ skipped (no NIM)`. Runs in about 1 minute on a CPU |
| Kaggle access | The same, on the real 100k-order Olist data |
| A NIM at `KUMO_NIM_URL` | Everything, including Kumo predictions, explanations, `predict_task` and write-back to SQLite |

| Env var | Default | Meaning |
|---|---|---|
| `OLIST_SOURCE` | `auto` | `kaggle` (fail if the download fails), `synthetic`, or `auto` (try Kaggle, then fall back) |
| `OLIST_DIR` | – | Folder with the 9 Olist CSVs, e.g. a zip downloaded in a browser |
| `KUMO_NIM_URL` | `http://localhost:8000` | NIM base URL |
| `KUMO_NIM_API_KEY` | – | Sent as `X-API-Key` when a gateway fronts the NIM |
| `KUMO_RUN_MODE` | `fast` | `debug` / `fast` / `normal` / `best` |
| `KUMO_TUTORIAL_WORK` | `./kumo_work` | Output folder for the CSVs, `olist.db` (our SQL), `olist_kumo*.db` (Kumo's copies), `predictions.db`, `scoreboard.csv` (git-ignored) |

---

## 2. How Kumo Relational predicts without training

```
          your SQLite file
                │  Graph.from_sqlite()      (tables + PK/FK links + time columns)
                ▼
   ┌────────── relational graph ──────────┐
   │ customers ─ orders ─ order_items ─ … │
   └──────────────────────────────────────┘
                │  PQL: "PREDICT COUNT(order_items.*, 0, 30, days) = 0 FOR EACH sellers.seller_id"
                ▼
  1. Context: sample historical entities at earlier anchor times and compute
     their TRUE labels from the data (100 / 1k / 5k / 10k examples, set by run_mode)
  2. Subgraphs: for each context and query entity, sample its multi-hop
     neighbourhood (orders → items → reviews …), using only rows older than its anchor
  3. One forward pass of the pre-trained model over [labelled context + unlabelled
     queries] → predictions as a pandas DataFrame. No gradient steps, no fitted weights.
```

The model is like an analyst who has seen thousands of databases. You hand it a few thousand
worked examples from *your* data, taken from *your* history, and it answers for new entities by
analogy. The examples change, but the model doesn't. So a new question is a new PQL string, not a
new project.

What this saves in practice (notebook section 9 shows the classic side in full):

| Step | Classic ML | Kumo Relational |
|---|---|---|
| Define the label | SQL per task, careful window logic | Part of the PQL string |
| Features | Hand-written multi-table aggregates (the notebook's seller snapshot is ~40 lines of SQL) | Read from the graph automatically, multi-hop |
| Leakage control | Every join filtered to `< anchor`, by hand | Built in: rows newer than the anchor are invisible |
| Training data | Stack monthly snapshots | Sampled automatically from history |
| Train / tune | Per task | None |
| New task | Repeat everything | Edit one line of PQL |

---

## 3. The dataset: Olist Brazilian E-Commerce

[Kaggle: `olistbr/brazilian-ecommerce`](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0). About 100k orders placed from 2016 to 2018 on Brazilian marketplaces, shipped as
9 normalized CSVs. It is one of the few freely available datasets that is truly *relational*:
customers, orders, items, payments, reviews, products, sellers, geolocation and a category
translation table.

Two quirks the notebook handles. Both are common in real systems:

* **`customer_id` is per order.** The real person is `customer_unique_id`. Key on the wrong one and
  every customer has exactly one order, so "will they come back?" becomes unlearnable.
* **`review_id` is not unique** across orders. The notebook builds a surrogate key
  (`review_id || '-' || order_id`).

Without Kaggle access, the notebook writes a **synthetic** dataset with the same 9 file names and
columns. It isn't noise. It plants the signals the real data has: seller quality drives lateness
and bad reviews and an earlier stop, sales fade before a seller churns, ~89% of customers buy once,
and most of a seller's catalog sits in one category. Results on it describe the generator, not Olist.

---

## 4. Designing the SQLite schema for Kumo

Kumo reads your normalized schema directly. These rules decide whether it reads it *well*:

| Rule | Why it matters |
|---|---|
| **One single-column primary key per table** | FKs must point at a PK. Add surrogate keys where the source has none (`order_item_key`, `payment_key`) |
| **FK name = PK name** (`orders.customer_id` → `customers.customer_id`) | Link inference matches on names. `REFERENCES` clauses are read too |
| **Every event table has a time column** | This is how the anchor time hides the future, and how `SUM(..., 0, 30, days)` is defined |
| **Split facts by when they become known** | A delivery date or review score on the order row would leak the future. The notebook moves them to `deliveries` and `reviews`, each with its own timestamp |
| **Put a time column on the table you aggregate** | `order_items` gets `purchased_at` (copied from `orders`) so seller revenue can be windowed |
| **Link the aggregated table directly to the entity** | PQL aggregates a one-hop neighbour. For "products a customer buys next", the notebook adds `order_items.customer_id` |
| **Store timestamps as ISO text** `YYYY-MM-DD HH:MM:SS` | Sorts and compares correctly in SQLite, and Kumo infers them as timestamps |
| **Don't run `ANALYZE` before `from_sqlite(path)`** | It creates `sqlite_stat1`, which zero-config discovery picks up as a table. Pass `tables=[...]` or skip it |
| **Give Kumo its own snapshot file** | See the warning below. The notebook queries `olist.db` and Kumo reads `olist_kumo.db` |

> **⚠️ Don't share one SQLite file between Kumo and Python's `sqlite3` in the same process.**
> Kumo's SQLite connector uses ADBC, which bundles its **own copy** of the SQLite library. During
> `predict()` it writes scratch tables (`tmp_*`, via `adbc_ingest`) into the database, inside a
> transaction it keeps open. Python's `sqlite3` is a second copy of SQLite in the same process. The
> two don't see each other's POSIX locks, so a `sqlite3` connection that merely **opens and closes**
> the file releases Kumo's locks. Kumo's open journal then looks "hot", and the next reader rolls it
> back under Kumo. This is case 2.2.1 of SQLite's
> [How To Corrupt An SQLite Database File](https://www.sqlite.org/howtocorrupt.html). While
> building the notebook it showed up as `database is locked`, then
> `attempt to write a readonly database` (read-only reader), then `database disk image is malformed`.
>
> What the notebook does: build `olist.db` with `sqlite3`, copy it once with the backup API to
> `olist_kumo.db`, and point every graph at the copy. The notebook's SQL only ever opens `olist.db`.
> Evaluation masks go into another Kumo-only copy, written *before* a graph opens it. Predictions
> go to `predictions.db`, which `ATTACH`es `olist.db` read-only. Use `contextlib.closing()`, because
> `with sqlite3.connect(...)` commits but does not close. Other **processes** are safe, since
> locking works normally across processes. Kumo needs write permission on its file, and other
> writers may see `SQLITE_BUSY` while a prediction runs. In production, point Kumo at a snapshot,
> not at the live application database.

The 8-table schema built in section 4 of the notebook:

```
customers 1──n orders 1──n order_items n──1 products
    ▲            │  │  │        │  │
    └────────────┼──┼──┼────────┘  └──n──1 sellers      (order_items.customer_id: direct link)
                 │  │  └──n payments   (paid_at)
                 │  └─────n deliveries (delivered_at, days_late)
                 └────────n reviews    (created_at, review_score, comment)
```

---

## 5. From SQLite to a graph

```python
from kumo_relational_client import relational

# Zero-config: every table, PKs, time columns, semantic types, links (declared FKs + name matching)
graph = relational.Graph.from_sqlite('olist.db')

# Explicit: what you commit to a repo
graph = relational.Graph.from_sqlite(
    'olist.db',
    tables=[{'name': 'orders', 'primary_key': 'order_id', 'time_column': 'purchased_at'}, ...],
    edges=[('orders', 'customer_id', 'customers'), ...],   # (src table, fkey, dst table)
)
graph['customers']['zip_prefix'].stype = 'categorical'      # an int that is really a category
graph.validate()
graph.metadata, graph.edges, graph['orders'].metadata        # inspect what was inferred
```

* `tables=[{'name': 'products', 'source_name': 'products_v2', ...}]` reads a different SQLite table
  (or a snapshot table) under the logical name used in PQL.
* `Graph.from_sqlite(path)` keeps a connection open for the life of the graph (there is no public
  `close()`), and the model handle caches the materialized graph. To pick up new data, write a fresh
  snapshot and build a new graph on it.
* **One live graph per SQLite file once you predict.** The open scratch-table transaction locks the
  file against a second Kumo connection (`database is locked`). Build schema variants before the
  first `predict()`, or point them at another snapshot, as the notebook does for the masked-label evaluation.
* `edges=[]` (empty) means "exactly these edges, no inference". `edges=None` means "infer".
* The same code works for other stores: `Graph.from_duckdb`, `from_postgres`, `from_snowflake`,
  `from_databricks`, or `from_data({...})` for pandas DataFrames.

---

## 6. PQL cheat-sheet

```
PREDICT <target> FOR EACH <entity_table>.<pk> [WHERE <entity filter>] [ASSUMING <condition>]
```

| Task | PQL used in the notebook | Output columns |
|---|---|---|
| Binary: seller churn | `PREDICT COUNT(order_items.*, 0, 30, days) = 0 FOR EACH sellers.seller_id WHERE COUNT(order_items.*, -90, 0, days) > 0` | `ENTITY, ANCHOR_TIMESTAMP, PREDICTION, FALSE_PROB, TRUE_PROB` |
| Regression: seller revenue | `PREDICT SUM(order_items.price, 0, 30, days) FOR EACH sellers.seller_id WHERE ...` | `ENTITY, ANCHOR_TIMESTAMP, PREDICTION` |
| Binary, rare positives: repurchase | `PREDICT COUNT(orders.*, 0, 90, days) > 0 FOR EACH customers.customer_id WHERE COUNT(orders.*, -365, 0, days) > 0` | as binary |
| Multiclass: missing category | `PREDICT products.category FOR EACH products.product_id` | one row **per entity per class**: `CLASS, SCORE, PREDICTED` |
| Ranking: next products | `PREDICT LIST_DISTINCT(order_items.product_id, 0, 90, days) RANK TOP 10 FOR EACH customers.customer_id` | *k* rows per entity: `CLASS, SCORE` |

* **Aggregations:** `SUM AVG MIN MAX COUNT COUNT_DISTINCT FIRST LAST LIST_DISTINCT`.
  Windows are `(start, end, unit)` relative to the anchor time. Negative values look back.
* **Filtered aggregates:** `COUNT(reviews.* WHERE reviews.review_score <= 2, 0, 30, days)`.
* **Pick entities with `indices=[...]`.** This overrides any ids written in the query. `FOR EACH`
  plus `indices` is the clearest style. Writing ids inline works too (`FOR users.user_id=1`, or
  `IN (1, 2)` with **two or more** values; a one-element `IN` list is a syntax error).
* **`WHERE` shapes the context, not the output.** It filters which historical entities become
  in-context examples. Every id in `indices` is scored regardless.
* **Join results on `ENTITY`**, never on row position: multiclass and ranking return several rows per entity.

The notebook's `check_pql()` validates a query against the graph **offline**, using the parser that
ships inside the installed engine. It caught every mistake above during authoring. It relies on an
internal function of engine 1.0.x, so treat it as a linter. The NIM has the final word.

---

## 7. `predict()` options that matter

```python
with RelationalClient(url='http://localhost:8000', timeout=300) as client:
    model = client.relational(graph)            # graph materialization is cached on the handle
    df = model.predict(query, indices=ids,
                       anchor_time=pd.Timestamp('2018-06-01'),   # backtest: model sees only older rows
                       run_mode='fast',                           # debug | fast | normal | best
                       batch_size='max',                          # split large id lists automatically
                       verbose=False)
```

| `run_mode` | In-context examples | Neighbours sampled per hop |
|---|---|---|
| `debug` | 100 | 16, 16, 4, 4, 1, 1 |
| `fast` (default) | 1,000 | 32, 32, 8, 8, 4, 4 |
| `normal` | 5,000 | 64, 64, 8, 8, 4, 4 |
| `best` | 10,000 | 64, 64, 8, 8, 4, 4 |

These values come from engine 1.0.2. Some articles mention `run_mode="accurate"`, but that value
does not exist in the client and is rejected with `INVALID_REQUEST`.

* `anchor_time`: `None` means "now", the latest timestamp in the data. `'entity'` uses each entity's
  own time column. `context_anchor_time` caps the context separately.
* Per-request limits: 1,000 entities (200 for ranking). `batch_size='max'` handles the split.
* `max_pq_iterations`: raise it when positives are rare or the `WHERE` filter is strict, so Kumo
  can collect enough labelled examples.
* `inference_config=dict(output_type='quantiles')` returns quantiles for regression.
* `explain=dict(skip_summary=True)` returns an `Explanation` for **one** entity (`.prediction`,
  `.details`). Without `skip_summary`, the `[explain]` extra plus `KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY`
  sends the entity's raw subgraph values to an external LLM (OpenAI by default) to write a summary.
* `predict_task(context=df, predict=df, task_type=..., entity_table=...)`: bring your own labels
  (`ENTITY, ANCHOR_TIMESTAMP, TARGET`) when the label isn't expressible in PQL. Section 16 feeds it the
  SQL snapshot built for the baseline.

---

## 8. Backtesting honestly

Every use case predicts **as of a past date** (`TEST_ANCHOR = 2018-06-01`, or `2018-05-01` for the
90-day horizon) and grades the prediction against what SQL says really happened afterwards. The
classic baseline trains only on monthly snapshots whose label windows end before the test anchor.
Use the same discipline for any model you compare. A baseline that peeks at the future makes
everything look bad by comparison.

Metric choice per task: AUC and average precision for churn and repurchase (accuracy means nothing
at 3% positives), MAE plus Spearman for heavy-tailed revenue, top-1 and top-3 accuracy for category,
hit@10 for recommendations.

### Reference numbers (synthetic data, CPU, no NIM)

These come from a full run of the notebook with `OLIST_SOURCE=synthetic`. Your Kaggle numbers will
differ. Kumo rows appear only when a NIM is attached, and none was available when this was written,
so **no Kumo results are claimed here**. Run it and fill in the scoreboard yourself.

| Use case | Naive rule | SQL features + GBM |
|---|---|---|
| 1 Seller churn (AUC) | 0.785 (days since last sale) | 0.838 |
| 2 Seller revenue (Spearman) | 0.655 (last 30 days repeats) | 0.686 |
| 3 Customer repurchase (AUC / AP, 3.5% positive) | 0.808 / 0.106 | 0.822 / 0.176 |
| 4 Product category (top-1 acc) | 0.310 (product attributes only) | 0.733 (plus a hand-built seller-mix feature) |
| 5 Next products (hit@10) | 0.020 (popular items) | – |

Use case 4 shows the point of a relational model. Product attributes alone are weak (0.31). The
signal is one hop away: what else does this seller sell? In classic ML you get 0.73 only if you
think to build that feature. Kumo walks `products → order_items → sellers → order_items → products`
on its own.

---

## 9. Gotchas found while building this

| Symptom | Cause / fix |
|---|---|
| A 9th table `sqlite_stat1` in the graph | `ANALYZE` created it. Skip `ANALYZE` or pass `tables=[...]` |
| `Array with single element is not supported` | `IN ('a')` → write `= 'a'`, or use `indices=['a']` |
| `Aggregation not supported on primary key columns` | `LIST_DISTINCT` must target an FK column (`order_items.product_id`), not a PK |
| Repurchase is unlearnable on Olist | You keyed on `customer_id`. Use `customer_unique_id` |
| Kumo can't aggregate `order_items` for customers | Not a direct link. Add `order_items.customer_id` (section 4, used in section 13) |
| `database is locked`, `attempt to write a readonly database`, `database disk image is malformed` | Python's `sqlite3` opened the same file a Kumo graph uses, in the same process. Give Kumo its own copy (section 4 warning) |
| Results misaligned with your ids | Join on `ENTITY`. Multiclass and ranking return several rows per entity |
| `run_mode must be one of 'debug', 'fast', 'normal', 'best'` | There is no `accurate` mode |
| Editing table data in place isn't picked up | The handle caches the materialized graph. Rebuild the graph (schema changes are detected) |
| `api_key` refused | Keys are refused over plain `http://` to non-local hosts. Use `https://` |

---

## 10. Running the weights locally instead of the NIM

NVIDIA also ships the model in [`structured-data-models`](https://github.com/NVIDIA/structured-data-models)
(`pip install structured-data-models`, Python 3.11+, PyTorch 2.7+, GPU recommended). It exposes
`sdm.models.KumoRelational` next to the tabular models (`TabICLv2`, `KumoTabular`). That library
takes tensors rather than a SQLite file, so you would do the graph sampling yourself. The NIM and
client path in this notebook handles sampling, context building and batching for you. See the
library docs for the exact tensor interface. This tutorial doesn't cover it because the interface
couldn't be checked without a GPU.

## 11. References

* Docs: [docs.nvidia.com/sdgm](https://docs.nvidia.com/sdgm/) · [Quickstart](https://docs.nvidia.com/sdgm/quick-start/rfm) · [Querying](https://docs.nvidia.com/sdgm/rfm/querying-rfm)
* Model and blog: [huggingface.co/nvidia/Kumo-Relational](https://huggingface.co/nvidia/Kumo-Relational) · [HF blog](https://huggingface.co/blog/nvidia/kumo-relational)
* NIM: [build.nvidia.com/nvidia/kumo-relational](https://build.nvidia.com/nvidia/kumo-relational)
* Client: [github.com/kumo-ai/kumo-relational-client](https://github.com/kumo-ai/kumo-relational-client) (this tutorial was written against `kumo-relational-client` 1.0.2)
* Data: [Olist on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

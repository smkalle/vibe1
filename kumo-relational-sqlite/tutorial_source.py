# %% [markdown]
# # NVIDIA Kumo Relational on SQLite: multi-table predictions without training
#
# This notebook is the hands-on companion to [`README.md`](README.md). You will:
#
# 1. Download the **Olist Brazilian E-Commerce** dataset from Kaggle (100k real orders across
#    9 CSV files). If Kaggle isn't reachable, the notebook builds a synthetic dataset with the
#    same schema.
# 2. Turn it into a **SQLite database** with proper primary/foreign keys. This is the shape your
#    production data usually already has.
# 3. Build a **Kumo Relational graph straight from the SQLite file** (`Graph.from_sqlite`).
# 4. Write **Predictive Query Language (PQL)** for five business questions:
#
#    | # | Question | Task type | Entity |
#    |---|----------|-----------|--------|
#    | 1 | Will this seller make **zero sales** in the next 30 days? | binary classification | `sellers` |
#    | 2 | How much **revenue** will this seller make in the next 30 days? | regression | `sellers` |
#    | 3 | Will this customer **buy again** within 90 days? | binary classification (rare positives) | `customers` |
#    | 4 | What is the **category** of this un-labelled product? | multiclass (static column) | `products` |
#    | 5 | Which **products** will this customer buy next? | ranking / link prediction | `customers` |
#
# 5. **Backtest** every prediction against what actually happened, and compare it with a
#    classic approach: SQL feature engineering plus gradient boosting, trained per task.
# 6. Write the predictions **back into SQLite**, where downstream apps can read them.
#
# **No GPU?** Everything except the Kumo calls runs on a laptop CPU: data prep, SQL, graph
# construction, offline PQL checks and the classic baselines. Kumo cells print `skipped` until a
# NIM endpoint answers at `KUMO_NIM_URL`.

# %% [markdown]
# ## 0. Setup
#
# ```bash
# pip install "kumo-relational-client[relational,sqlite]" scikit-learn matplotlib kagglehub jupyter
# ```
#
# Start a Kumo Relational NIM. You need an NVIDIA GPU such as an A10G, L4, L40S or H100, plus an
# NGC API key:
#
# ```bash
# docker login nvcr.io          # user: $oauthtoken   password: <NGC API key>
# docker run -it --rm --gpus all --shm-size=16GB -p 8000:8000 nvcr.io/nim/nvidia/kumo-relational:1.0.0
# ```
#
# All settings come from environment variables, so the same notebook runs on a laptop, a GPU box or in CI:
#
# | Variable | Default | Meaning |
# |---|---|---|
# | `OLIST_SOURCE` | `auto` | `kaggle` (fail if the download fails), `synthetic`, or `auto` (try Kaggle, fall back to synthetic) |
# | `OLIST_DIR` | – | Folder that already holds the 9 Olist CSVs (e.g. a manual Kaggle download) |
# | `KUMO_NIM_URL` | `http://localhost:8000` | Kumo Relational NIM base URL |
# | `KUMO_NIM_API_KEY` | – | Only if an authenticating gateway fronts the NIM (sent as `X-API-Key`) |
# | `KUMO_RUN_MODE` | `fast` | `debug` / `fast` / `normal` / `best`: 100 / 1k / 5k / 10k in-context examples |
# | `KUMO_TUTORIAL_WORK` | `./kumo_work` | Where the CSVs, the SQLite DB and outputs go |

# %%
import os
import sqlite3
from contextlib import closing
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 160)

WORK = Path(os.getenv('KUMO_TUTORIAL_WORK', 'kumo_work')).resolve()
WORK.mkdir(parents=True, exist_ok=True)
DB_PATH = WORK / 'olist.db'                # read by our SQL (Python's sqlite3)
KUMO_DB_PATH = WORK / 'olist_kumo.db'      # read by Kumo only: a snapshot of olist.db (see section 4)

DATA_SOURCE = os.getenv('OLIST_SOURCE', 'auto')
OLIST_DIR = os.getenv('OLIST_DIR')
NIM_URL = os.getenv('KUMO_NIM_URL', 'http://localhost:8000')
NIM_API_KEY = os.getenv('KUMO_NIM_API_KEY') or None
RUN_MODE = os.getenv('KUMO_RUN_MODE', 'fast')
SEED = 7

# Chart styling: thin marks, recessive axes, one accent per series.
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
plt.rcParams.update({
    'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'axes.titlecolor': INK,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.axisbelow': True, 'figure.dpi': 110,
})
print('work dir :', WORK)
print('NIM url  :', NIM_URL, '| run_mode:', RUN_MODE)

# %% [markdown]
# ## 1. Warm-up: the two-table example from the announcement, in SQLite
#
# Before using real data, put the `users` + `orders` example from the announcement into a SQLite file
# and let Kumo read it. Notice two things:
#
# * The `REFERENCES` clause is the foreign key. `Graph.from_sqlite` reads FKs that the catalog
#   declares, and it **also infers** links from matching key names. So `orders.user_id` would be
#   linked to `users.user_id` even without the DDL.
# * Timestamps are stored as ISO-8601 text (`YYYY-MM-DD HH:MM:SS`). This is the most common
#   format in SQLite, and Kumo detects these columns as time columns.

# %%
from kumo_relational_client import relational

toy_db = WORK / 'toy.db'
toy_db.unlink(missing_ok=True)
with closing(sqlite3.connect(toy_db)) as con:   # closing(): `with sqlite3.connect()` alone never closes
    con.executescript("""
        CREATE TABLE users (
            user_id     INTEGER PRIMARY KEY,
            signup_date TEXT,
            plan        TEXT
        );
        CREATE TABLE orders (
            order_id  INTEGER PRIMARY KEY,
            user_id   INTEGER REFERENCES users(user_id),
            price     REAL,
            timestamp TEXT
        );
        INSERT INTO users VALUES
            (1, '2024-01-01', 'free'), (2, '2024-02-15', 'pro'),
            (3, '2024-03-20', 'free'), (4, '2024-04-10', 'pro');
        INSERT INTO orders VALUES
            (101, 1, 50.0,  '2025-01-10'), (102, 1, 30.0, '2025-02-15'),
            (103, 2, 100.0, '2025-01-20'), (104, 3, 75.0, '2025-03-05'),
            (105, 1, 40.0,  '2025-04-01');
    """)

toy_graph = relational.Graph.from_sqlite(str(toy_db))   # prints inferred metadata + links
toy_graph.validate()

# %% [markdown]
# `users` has no time column, so it is treated as a static entity table. `orders` has one
# (`timestamp`), so it is an event table, and `SUM(orders.price, 0, 30, days)` can be computed
# relative to any anchor time. That is the whole mental model, and the rest of the notebook
# applies it to 8 tables and ~100k orders.

# %% [markdown]
# ## 2. Get the data: Olist Brazilian E-Commerce (Kaggle)
#
# [`olistbr/brazilian-ecommerce`](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) is a
# public dataset (CC BY-NC-SA 4.0) of ~100k orders placed on Brazilian marketplaces from 2016 to 2018.
# It is one of the best freely available *relational* datasets because it ships as 9 normalized tables:
#
# ```
# customers ──< orders ──< order_items >── products
#                  │            │
#                  ├──< payments └──> sellers
#                  └──< reviews
# ```
#
# `kagglehub` downloads it once and caches it. It needs a Kaggle account: put your API token in
# `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME`/`KAGGLE_KEY`. If that isn't possible, download
# the zip from the dataset page in a browser and set `OLIST_DIR` to the unzipped folder.
#
# With `OLIST_SOURCE=auto` and no Kaggle access, the notebook writes a **synthetic** dataset instead.
# It has the same 9 file names, the same columns and similar statistics, so the whole notebook runs offline.
# The numbers you see then describe the synthetic data, not Olist.

# %%
OLIST_FILES = {
    'customers': 'olist_customers_dataset.csv',
    'orders': 'olist_orders_dataset.csv',
    'items': 'olist_order_items_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv',   # not used below
    'category_en': 'product_category_name_translation.csv',
}


def fetch_kaggle_olist() -> Path | None:
    """Returns the folder holding the Olist CSVs, or None to use synthetic data."""
    if OLIST_DIR:
        return Path(OLIST_DIR)
    if DATA_SOURCE == 'synthetic':
        return None
    try:
        import kagglehub
        path = Path(kagglehub.dataset_download('olistbr/brazilian-ecommerce'))
        print('Kaggle dataset cached at', path)
        return path
    except Exception as error:   # no kagglehub, no credentials, no network...
        if DATA_SOURCE == 'kaggle':
            raise
        print(f'Kaggle download unavailable ({type(error).__name__}: {str(error)[:120]})')
        print('-> falling back to the synthetic Olist-shaped dataset')
        return None


olist_dir = fetch_kaggle_olist()

# %% [markdown]
# ### 2b. Synthetic fallback with the same schema as Olist
#
# Skip this cell's details if you are using the real data. The generator is not random noise. It
# plants the kinds of signal the real data has, so the models have something to learn:
#
# * **Sellers** have a hidden quality score and a lifetime. Low-quality sellers ship late, collect
#   bad reviews, and stop selling sooner. Their volume also fades in the 60 days before they stop.
# * **Customers** mostly buy once. A few repeat, more often after a good review, and often from the same seller.
# * **Products** belong to a seller, and most of a seller's catalog sits in one category. Price,
#   weight and photo count depend on the category.

# %%
CATEGORIES = [  # (portuguese, english, median price R$, median weight g, photos)
    ('cama_mesa_banho', 'bed_bath_table', 80, 1500, 2), ('beleza_saude', 'health_beauty', 90, 500, 2),
    ('esporte_lazer', 'sports_leisure', 85, 900, 2), ('moveis_decoracao', 'furniture_decor', 95, 3000, 1),
    ('informatica_acessorios', 'computers_accessories', 110, 600, 2), ('utilidades_domesticas', 'housewares', 70, 1800, 1),
    ('relogios_presentes', 'watches_gifts', 200, 350, 3), ('telefonia', 'telephony', 45, 200, 2),
    ('ferramentas_jardim', 'garden_tools', 100, 2500, 1), ('automotivo', 'auto', 120, 1600, 2),
    ('brinquedos', 'toys', 75, 900, 3), ('cool_stuff', 'cool_stuff', 160, 1700, 3),
    ('perfumaria', 'perfumery', 95, 400, 1), ('bebes', 'baby', 110, 2000, 2),
    ('eletronicos', 'electronics', 60, 300, 2), ('papelaria', 'stationery', 50, 800, 1),
    ('fashion_bolsas_e_acessorios', 'fashion_bags_accessories', 55, 400, 4), ('pet_shop', 'pet_shop', 70, 1800, 1),
    ('moveis_escritorio', 'office_furniture', 180, 10000, 1), ('consoles_games', 'consoles_games', 140, 600, 2),
]
STATES = {'SP': ('sao paulo', 1000, .42), 'RJ': ('rio de janeiro', 20000, .13), 'MG': ('belo horizonte', 30000, .12),
          'RS': ('porto alegre', 90000, .055), 'PR': ('curitiba', 80000, .05), 'SC': ('florianopolis', 88000, .037),
          'BA': ('salvador', 40000, .034), 'DF': ('brasilia', 70000, .022), 'GO': ('goiania', 74000, .02),
          'ES': ('vitoria', 29000, .02)}
GOOD_COMMENTS = ['Produto chegou antes do prazo, recomendo.', 'Ótimo vendedor, tudo certo.', 'Muito bom, conforme anunciado.']
BAD_COMMENTS = ['Ainda não recebi o produto.', 'Produto veio com defeito.', 'Entrega muito atrasada, não recomendo.']
TS = '%Y-%m-%d %H:%M:%S'


def make_synthetic_olist(out_dir: Path, seed: int = SEED, n_customers: int = 12_000,
                         n_sellers: int = 450, n_products: int = 4_000) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    hexid = lambda: rng.bytes(16).hex()                                   # noqa: E731
    start, end = pd.Timestamp('2016-10-01'), pd.Timestamp('2018-08-31')
    span = (end - start).days
    state_codes = list(STATES)
    state_p = np.array([STATES[s][2] for s in state_codes]); state_p /= state_p.sum()
    cat_p = np.linspace(2.0, 0.5, len(CATEGORIES)); cat_p /= cat_p.sum()

    def place(n):
        st = rng.choice(state_codes, n, p=state_p)
        return st, [STATES[s][0] for s in st], [STATES[s][1] + int(rng.integers(0, 999)) for s in st]

    # Sellers: quality, popularity, active window, main category.
    s_id = [hexid() for _ in range(n_sellers)]
    s_q = rng.normal(0, 1, n_sellers)
    s_pop = rng.lognormal(0, 1.0, n_sellers)
    s_start = start + pd.to_timedelta(rng.integers(0, int(span * .8), n_sellers), unit='D')
    s_end = s_start + pd.to_timedelta(rng.exponential(420 * np.exp(.45 * s_q)) + 30, unit='D')
    s_cat = rng.choice(len(CATEGORIES), n_sellers, p=cat_p)
    s_state, s_city, s_zip = place(n_sellers)

    # Products: every seller gets at least one; 75% share the seller's main category.
    p_seller = np.concatenate([np.arange(n_sellers),
                               rng.choice(n_sellers, n_products - n_sellers, p=s_pop / s_pop.sum())])
    p_cat = np.where(rng.random(n_products) < .75, s_cat[p_seller],
                     rng.choice(len(CATEGORIES), n_products, p=cat_p))
    p_id = [hexid() for _ in range(n_products)]
    p_price = np.array([CATEGORIES[c][2] for c in p_cat]) * rng.lognormal(0, .5, n_products)
    p_weight = np.array([CATEGORIES[c][3] for c in p_cat]) * rng.lognormal(0, .6, n_products)
    p_photos = np.array([CATEGORIES[c][4] for c in p_cat]) + rng.poisson(.6, n_products)
    p_pop = rng.lognormal(0, 1.0, n_products)
    seller_products = {s: np.flatnonzero(p_seller == s) for s in range(n_sellers)}

    # Customers and their orders.
    day_w = .25 + np.arange(span) / span; day_w /= day_w.sum()          # marketplace growth
    c_rep = rng.beta(1.0, 9.0, n_customers)                               # repeat propensity
    c_state, c_city, c_zip = place(n_customers)
    customers, orders, items, payments, reviews = [], [], [], [], []

    for c in range(n_customers):
        uid = hexid()
        t = start + pd.Timedelta(days=int(rng.choice(span, p=day_w)), seconds=int(rng.integers(0, 86400)))
        last_seller = None
        while t < end:
            active = np.flatnonzero((s_start <= t) & (s_end > t))
            if len(active) == 0:
                break
            if last_seller in active and rng.random() < .5:
                s = last_seller
            else:
                days_left = (s_end[active] - t).days
                w = s_pop[active] * np.where(days_left < 60, .35, 1.0)    # fading before churn
                s = int(rng.choice(active, p=w / w.sum()))
            oid, cid = hexid(), hexid()
            customers.append((cid, uid, c_zip[c], c_city[c], c_state[c]))
            canceled = rng.random() < .012
            est_days = int(rng.integers(15, 35))
            ship_days = max(2.0, rng.normal(11 - 2.5 * s_q[s], 5))
            delivered = None if canceled else t + pd.Timedelta(days=ship_days)
            estimated = (t + pd.Timedelta(days=est_days)).normalize()
            orders.append((oid, cid, 'canceled' if canceled else 'delivered', t.strftime(TS),
                           (t + pd.Timedelta(minutes=int(rng.integers(5, 600)))).strftime(TS),
                           None if canceled else (t + pd.Timedelta(days=2)).strftime(TS),
                           None if canceled else delivered.strftime(TS), estimated.strftime(TS)))
            prods = seller_products[s]
            n_items = 1 + (rng.random() < .1) + (rng.random() < .03)
            chosen = rng.choice(prods, size=n_items, p=p_pop[prods] / p_pop[prods].sum())
            total = 0.0
            for k, p in enumerate(chosen, start=1):
                price = round(float(p_price[p]), 2)
                freight = round(float(8 + p_weight[p] / 400 + rng.gamma(2, 3)), 2)
                total += price + freight
                items.append((oid, k, p_id[p], s_id[s], (t + pd.Timedelta(days=6)).strftime(TS), price, freight))
            ptype = rng.choice(['credit_card', 'boleto', 'voucher', 'debit_card'], p=[.74, .19, .055, .015])
            inst = int(rng.integers(1, 11)) if ptype == 'credit_card' else 1
            payments.append((oid, 1, ptype, inst, round(total, 2)))
            score = 3
            if not canceled or rng.random() < .5:
                late = delivered is not None and delivered > estimated
                raw = 4.4 + .35 * s_q[s] - 2.2 * late - .4 * (ship_days > 20) + rng.normal(0, .9)
                score = int(np.clip(round(raw), 1, 5)) if not canceled else 1
                comment = None
                if rng.random() < (.7 if score <= 2 else .3):
                    comment = rng.choice(BAD_COMMENTS if score <= 2 else GOOD_COMMENTS)
                created = (delivered or estimated) + pd.Timedelta(days=1)
                reviews.append((hexid(), oid, score, None, comment, created.normalize().strftime(TS),
                                (created + pd.Timedelta(days=2)).strftime(TS)))
            last_seller = s
            p_next = c_rep[c] * (1.6 if score >= 4 else .5)
            if rng.random() >= p_next:
                break
            t = t + pd.Timedelta(days=float(3 + rng.exponential(70)))

    write = lambda key, rows, cols: pd.DataFrame(rows, columns=cols).to_csv(out_dir / OLIST_FILES[key], index=False)  # noqa: E731
    write('customers', customers, ['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state'])
    write('orders', orders, ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at',
                             'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
    write('items', items, ['order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value'])
    write('payments', payments, ['order_id', 'payment_sequential', 'payment_type', 'payment_installments', 'payment_value'])
    write('reviews', reviews, ['review_id', 'order_id', 'review_score', 'review_comment_title', 'review_comment_message',
                               'review_creation_date', 'review_answer_timestamp'])
    missing = rng.random(n_products) < .02                                # Olist has ~1.9% unlabeled products
    write('products', [(p_id[i], None if missing[i] else CATEGORIES[p_cat[i]][0], int(rng.integers(20, 64)),
                        int(rng.lognormal(6.4, .7) * (1.5 if p_cat[i] in (4, 14, 19) else 1)), int(p_photos[i]),
                        int(p_weight[i]), int(15 + p_weight[i] ** .33), int(5 + p_weight[i] ** .3), int(10 + p_weight[i] ** .3))
                       for i in range(n_products)],
          ['product_id', 'product_category_name', 'product_name_lenght', 'product_description_lenght',
           'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'])
    write('sellers', list(zip(s_id, s_zip, s_city, s_state)), ['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state'])
    write('category_en', [c[:2] for c in CATEGORIES], ['product_category_name', 'product_category_name_english'])
    return out_dir


if olist_dir is None:
    t0 = time.perf_counter()
    olist_dir = make_synthetic_olist(WORK / 'olist_synthetic')
    print(f'synthetic Olist written to {olist_dir} in {time.perf_counter() - t0:.0f}s')
IS_SYNTHETIC = 'olist_synthetic' in str(olist_dir)

# %% [markdown]
# ## 3. Look at the raw CSVs
#
# Two Olist quirks matter for relational modelling. Both are handled in the next step:
#
# * **`customer_id` is per order.** Olist issues a new `customer_id` for every order, and the real
#   person is `customer_unique_id`. If you keyed on `customer_id`, every customer would have exactly
#   one order and "will they buy again?" could never be learned.
# * **`review_id` is not unique.** The same review id appears on several orders, so we build a
#   surrogate key.

# %%
raw = {k: pd.read_csv(olist_dir / f, encoding='utf-8-sig') for k, f in OLIST_FILES.items() if k != 'geolocation' and (olist_dir / f).exists()}
display(pd.DataFrame({'rows': {k: len(v) for k, v in raw.items()},
                      'columns': {k: len(v.columns) for k, v in raw.items()}}))
c = raw['customers']
print(f"customer_id values: {c.customer_id.nunique():,} | real customers (customer_unique_id): {c.customer_unique_id.nunique():,}")
print(f"duplicate review_id rows: {raw['reviews'].review_id.duplicated().sum():,}")
raw['orders'].head(3)

# %% [markdown]
# ## 4. Design the relational schema and load it into SQLite
#
# Kumo works with your normalized schema directly. You don't flatten anything. A few design rules
# carry most of the weight:
#
# | Rule | Why | Where it shows up here |
# |---|---|---|
# | Every table gets a **single-column primary key** | Rows must be addressable, and FKs point at PKs | Surrogate keys `order_item_key`, `payment_key`, `review_key`, `delivery_id` |
# | **FK column name = PK column name** | Link inference matches on names (`orders.customer_id` → `customers.customer_id`) | All FKs |
# | Every **event** table has a **time column** | Kumo hides rows newer than the anchor time from the model. That is how it avoids leakage, and how `SUM(..., 0, 30, days)` is defined | `purchased_at`, `created_at`, `delivered_at` |
# | **Split facts by when they become known** | Delivery date and review score are unknown at purchase time. If they sat on the order row, they would leak the future | `deliveries` and `reviews` are separate tables with their own timestamps |
# | Put a time column on the table you **aggregate** | PQL targets aggregate one table, so it needs its own timestamp | `order_items.purchased_at` (copied from `orders`) |
# | **Link the aggregated table directly to the entity** | PQL aggregates a table one hop from the entity. `order_items` hangs off `orders`, so "products a customer buys" needs a direct key | `order_items.customer_id` (copied from `orders`, used in use case 5) |
# | **Give Kumo its own snapshot file** | See the warning below | `olist.db` for our SQL, `olist_kumo.db` for Kumo |
#
# > **⚠️ Don't share one SQLite file between Kumo and `sqlite3` in the same Python process.**
# > Kumo's SQLite connector uses ADBC, which bundles its **own copy** of the SQLite library. During
# > `predict()` it writes scratch tables (`tmp_*`) into the database inside a transaction it keeps
# > open. Python's `sqlite3` is a second copy of SQLite. Two copies in one process don't see each
# > other's POSIX locks, so a `sqlite3` connection that merely opens and closes the same file drops
# > Kumo's locks. That is case 2.2.1 of SQLite's [How To Corrupt](https://www.sqlite.org/howtocorrupt.html)
# > page. While building this notebook it produced `database is locked`,
# > `attempt to write a readonly database` and finally `database disk image is malformed`.
# > The fix is simple. After the build, copy the database once (`olist_kumo.db`) and let Kumo alone
# > open it. Our SQL keeps using `olist.db`. Other *processes* are safe, because locking works
# > normally across processes. Kumo still needs write permission on its file for the scratch tables.
#
# Resulting graph (8 tables):
#
# ```
#                 ┌──────────── customers ────────────┐
#                 │ customer_id (PK)                  │
#                 └───────────────┬───────────────────┘
#                                 │ 1:n
#    deliveries ─┐          ┌── orders ──┐          ┌─ reviews
#    (delivered_at)  n:1    │ purchased_at│   n:1    (created_at)
#                 └────────>│ order_id PK │<────────┘
#    payments ─────────────>└─────┬──────┘
#                                 │ 1:n
#    products <──────────── order_items ────────────> sellers
#    product_id PK            purchased_at            seller_id PK
#                             customer_id ──> customers (direct link for use case 5)
# ```

# %%
DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE customers (
    customer_id   TEXT PRIMARY KEY,          -- = Olist customer_unique_id (the real person)
    zip_prefix    INTEGER,
    city          TEXT,
    state         TEXT
);
CREATE TABLE sellers (
    seller_id     TEXT PRIMARY KEY,
    zip_prefix    INTEGER,
    city          TEXT,
    state         TEXT
);
CREATE TABLE products (
    product_id         TEXT PRIMARY KEY,
    category           TEXT,                 -- English name; NULL for ~2% of products
    name_length        INTEGER,
    description_length INTEGER,
    photos_qty         INTEGER,
    weight_g           REAL,
    volume_cm3         REAL
);
CREATE TABLE orders (
    order_id                TEXT PRIMARY KEY,
    customer_id             TEXT NOT NULL REFERENCES customers(customer_id),
    status                  TEXT,
    purchased_at            TEXT NOT NULL,   -- time column
    estimated_delivery_days INTEGER          -- promised at checkout, so known at purchase time
);
CREATE TABLE order_items (
    order_item_key TEXT PRIMARY KEY,         -- order_id || '-' || order_item_id
    order_id       TEXT NOT NULL REFERENCES orders(order_id),
    product_id     TEXT NOT NULL REFERENCES products(product_id),
    seller_id      TEXT NOT NULL REFERENCES sellers(seller_id),
    price          REAL,
    freight_value  REAL,
    purchased_at   TEXT NOT NULL,            -- copied from orders so items can be aggregated over time
    customer_id    TEXT NOT NULL REFERENCES customers(customer_id)  -- copied from orders: direct link
);
CREATE TABLE payments (
    payment_key   TEXT PRIMARY KEY,
    order_id      TEXT NOT NULL REFERENCES orders(order_id),
    payment_type  TEXT,
    installments  INTEGER,
    payment_value REAL,
    paid_at       TEXT NOT NULL
);
CREATE TABLE deliveries (
    delivery_id  TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL REFERENCES orders(order_id),
    delivered_at TEXT NOT NULL,              -- known only once delivered
    days_late    REAL                        -- delivered minus promised date (>0 = late)
);
CREATE TABLE reviews (
    review_key   TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL REFERENCES orders(order_id),
    review_score INTEGER,
    comment      TEXT,                       -- Portuguese free text (often empty)
    created_at   TEXT NOT NULL
);
CREATE INDEX ix_orders_customer   ON orders(customer_id, purchased_at);
CREATE INDEX ix_items_seller      ON order_items(seller_id, purchased_at);
CREATE INDEX ix_items_order       ON order_items(order_id);
CREATE INDEX ix_items_product     ON order_items(product_id);
CREATE INDEX ix_items_customer    ON order_items(customer_id, purchased_at);
CREATE INDEX ix_payments_order    ON payments(order_id);
CREATE INDEX ix_deliveries_order  ON deliveries(order_id);
CREATE INDEX ix_reviews_order     ON reviews(order_id);
"""


def ts(series: pd.Series) -> pd.Series:
    """Olist timestamps -> ISO text that SQLite compares correctly."""
    return pd.to_datetime(series, errors='coerce').dt.strftime(TS)


def build_sqlite(raw: dict[str, pd.DataFrame], db_path: Path) -> None:
    cust, o = raw['customers'], raw['orders'].copy()
    uid = dict(zip(cust.customer_id, cust.customer_unique_id))

    customers = (cust.drop_duplicates('customer_unique_id')
                 .rename(columns={'customer_unique_id': 'customer_id_', 'customer_zip_code_prefix': 'zip_prefix',
                                  'customer_city': 'city', 'customer_state': 'state'})
                 .drop(columns='customer_id').rename(columns={'customer_id_': 'customer_id'}))

    o['purchased'] = pd.to_datetime(o.order_purchase_timestamp)
    o = o[o.purchased < '2018-09-01']                    # Olist tails off after Aug 2018
    est = pd.to_datetime(o.order_estimated_delivery_date)
    dlv = pd.to_datetime(o.order_delivered_customer_date, errors='coerce')
    orders = pd.DataFrame({'order_id': o.order_id, 'customer_id': o.customer_id.map(uid),
                           'status': o.order_status, 'purchased_at': o.purchased.dt.strftime(TS),
                           'estimated_delivery_days': (est - o.purchased).dt.days})
    deliveries = pd.DataFrame({'delivery_id': 'd-' + o.order_id, 'order_id': o.order_id,
                               'delivered_at': dlv.dt.strftime(TS),
                               'days_late': ((dlv - est).dt.total_seconds() / 86400).round(2)}).dropna(subset=['delivered_at'])
    when = dict(zip(orders.order_id, orders.purchased_at))

    it = raw['items'][raw['items'].order_id.isin(when)]
    order_items = pd.DataFrame({'order_item_key': it.order_id + '-' + it.order_item_id.astype(str),
                                'order_id': it.order_id, 'product_id': it.product_id, 'seller_id': it.seller_id,
                                'price': it.price, 'freight_value': it.freight_value,
                                'purchased_at': it.order_id.map(when),
                                'customer_id': it.order_id.map(dict(zip(orders.order_id, orders.customer_id)))})

    pay = raw['payments'][raw['payments'].order_id.isin(when)]
    payments = pd.DataFrame({'payment_key': pay.order_id + '-' + pay.payment_sequential.astype(str),
                             'order_id': pay.order_id, 'payment_type': pay.payment_type,
                             'installments': pay.payment_installments, 'payment_value': pay.payment_value,
                             'paid_at': pay.order_id.map(when)})

    rv = raw['reviews'][raw['reviews'].order_id.isin(when)]
    reviews = pd.DataFrame({'review_key': rv.review_id + '-' + rv.order_id, 'order_id': rv.order_id,
                            'review_score': rv.review_score, 'comment': rv.review_comment_message,
                            'created_at': ts(rv.review_creation_date)}).drop_duplicates('review_key')

    en = dict(zip(raw['category_en'].product_category_name, raw['category_en'].product_category_name_english))
    pr = raw['products']
    products = pd.DataFrame({'product_id': pr.product_id,
                             'category': pr.product_category_name.map(lambda v: en.get(v, v) if isinstance(v, str) else None),
                             'name_length': pr.product_name_lenght, 'description_length': pr.product_description_lenght,
                             'photos_qty': pr.product_photos_qty, 'weight_g': pr.product_weight_g,
                             'volume_cm3': pr.product_length_cm * pr.product_height_cm * pr.product_width_cm})
    se = raw['sellers']
    sellers = pd.DataFrame({'seller_id': se.seller_id, 'zip_prefix': se.seller_zip_code_prefix,
                            'city': se.seller_city, 'state': se.seller_state})

    db_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(db_path)) as con:
        con.executescript(DDL)
        # Parents before children, so the FOREIGN KEY checks pass.
        for name, df in [('customers', customers), ('sellers', sellers), ('products', products),
                         ('orders', orders), ('order_items', order_items), ('payments', payments),
                         ('deliveries', deliveries), ('reviews', reviews)]:
            df.to_sql(name, con, if_exists='append', index=False, chunksize=20_000)
        con.commit()
        # No ANALYZE here: it creates sqlite_stat1, which from_sqlite() would pick up as a 9th table.


t0 = time.perf_counter()
build_sqlite(raw, DB_PATH)
print(f'built {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.1f}s')


def snapshot(src: Path, dst: Path) -> Path:
    """Copies a SQLite file with the online backup API (reads src, writes a fresh dst)."""
    dst.unlink(missing_ok=True)     # new inode: a graph still holding an old copy is never written under
    source, target = sqlite3.connect(f'{src.as_uri()}?mode=ro', uri=True), sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close(); source.close()
    return dst


snapshot(DB_PATH, KUMO_DB_PATH)
print('Kumo snapshot ->', KUMO_DB_PATH)

# %% [markdown]
# A small helper runs **read-only** SQL against `olist.db` and returns a DataFrame. From here on
# the notebook doesn't modify `olist.db`, and Python's `sqlite3` never opens `olist_kumo.db`.

# %%
def read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)


def sql(query: str, db: Path = DB_PATH, **params) -> pd.DataFrame:
    con = read_only(db)
    try:
        return pd.read_sql_query(query, con, params=params)
    finally:
        con.close()


sql("""
SELECT 'customers' AS table_name, COUNT(*) AS rows FROM customers UNION ALL
SELECT 'sellers', COUNT(*) FROM sellers         UNION ALL
SELECT 'products', COUNT(*) FROM products       UNION ALL
SELECT 'orders', COUNT(*) FROM orders           UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items UNION ALL
SELECT 'payments', COUNT(*) FROM payments       UNION ALL
SELECT 'deliveries', COUNT(*) FROM deliveries   UNION ALL
SELECT 'reviews', COUNT(*) FROM reviews
""")

# %% [markdown]
# ## 5. Explore with SQL: what signal is in the relations?
#
# Before predicting anything, check that the signal lives *across* tables. Doing that check
# without a model is the strength of SQL.

# %%
monthly = sql("""
SELECT substr(purchased_at, 1, 7) AS month, COUNT(*) AS orders,
       COUNT(DISTINCT customer_id) AS customers
FROM orders GROUP BY month ORDER BY month
""")
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(pd.to_datetime(monthly.month), monthly.orders, color=SERIES[0], linewidth=2)
ax.set_title('Orders per month')
ax.set_ylabel('orders')
ax.axvline(pd.Timestamp('2018-06-01'), color=MUTED, linewidth=1, linestyle='--')
ax.annotate('backtest anchor', (pd.Timestamp('2018-06-01'), ax.get_ylim()[1] * .92),
            xytext=(4, 0), textcoords='offset points', color=MUTED, fontsize=8)
plt.show()

# %%
# Lateness (deliveries) -> satisfaction (reviews): two event tables joined through orders.
sql("""
SELECT CASE WHEN d.days_late <= -7 THEN '1. >7 days early'
            WHEN d.days_late <= 0  THEN '2. on time'
            WHEN d.days_late <= 7  THEN '3. up to 7 days late'
            ELSE                        '4. >7 days late' END AS delivery,
       COUNT(*) AS reviews,
       ROUND(AVG(r.review_score), 2) AS avg_score,
       ROUND(100.0 * AVG(r.review_score <= 2), 1) AS pct_1_or_2_stars
FROM reviews r JOIN deliveries d USING (order_id)
GROUP BY delivery ORDER BY delivery
""")

# %%
# How rare is a repeat purchase? This sets expectations for use case 3.
sql("""
WITH per_customer AS (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id)
SELECT CASE WHEN n >= 4 THEN '4+' ELSE CAST(n AS TEXT) END AS orders_per_customer,
       COUNT(*) AS customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM per_customer), 2) AS pct
FROM per_customer GROUP BY orders_per_customer ORDER BY orders_per_customer
""")

# %%
# Seller concentration: revenue is heavy-tailed, so expect a skewed regression target.
sql("""
WITH rev AS (SELECT seller_id, SUM(price) AS revenue FROM order_items GROUP BY seller_id),
ranked AS (SELECT revenue, NTILE(10) OVER (ORDER BY revenue DESC) AS decile FROM rev)
SELECT decile, COUNT(*) AS sellers, ROUND(SUM(revenue)) AS revenue,
       ROUND(100.0 * SUM(revenue) / (SELECT SUM(revenue) FROM rev), 1) AS pct_of_total
FROM ranked GROUP BY decile ORDER BY decile
""")

# %% [markdown]
# ## 6. Build the Kumo graph from SQLite
#
# ### 6a. Zero-config: let Kumo infer everything
#
# `Graph.from_sqlite(path)` opens the file and discovers every table. It then infers each table's
# **primary key**, **time column** and **semantic types** (numerical, categorical, text, ID,
# timestamp) from samples. Finally it **links** tables using the declared `REFERENCES` plus
# name matching. Nothing is copied into memory up front: Kumo samples the SQLite file when it
# builds context.

# %%
graph = relational.Graph.from_sqlite(str(KUMO_DB_PATH))
graph.validate()

# %% [markdown]
# ### 6b. Explicit config: what you would commit to a repo
#
# Inference is convenient, but production code should pin the schema. Then a new column, or a
# sample where a key happens to look non-unique, can't silently change the graph. You can also
# override semantic types. For example, `zip_prefix` is an integer, but it is a *category*, not a
# quantity.

# %%
TABLES = [
    {'name': 'customers',   'primary_key': 'customer_id'},
    {'name': 'sellers',     'primary_key': 'seller_id'},
    {'name': 'products',    'primary_key': 'product_id'},
    {'name': 'orders',      'primary_key': 'order_id',       'time_column': 'purchased_at'},
    {'name': 'order_items', 'primary_key': 'order_item_key', 'time_column': 'purchased_at'},
    {'name': 'payments',    'primary_key': 'payment_key',    'time_column': 'paid_at'},
    {'name': 'deliveries',  'primary_key': 'delivery_id',    'time_column': 'delivered_at'},
    {'name': 'reviews',     'primary_key': 'review_key',     'time_column': 'created_at'},
]
EDGES = [  # (source table, foreign key, destination table)
    ('orders', 'customer_id', 'customers'),
    ('order_items', 'order_id', 'orders'),
    ('order_items', 'product_id', 'products'),
    ('order_items', 'seller_id', 'sellers'),
    ('order_items', 'customer_id', 'customers'),
    ('payments', 'order_id', 'orders'),
    ('deliveries', 'order_id', 'orders'),
    ('reviews', 'order_id', 'orders'),
]


def make_graph(db: Path = KUMO_DB_PATH, tables=TABLES, edges=EDGES, verbose=False):
    g = relational.Graph.from_sqlite(str(db), tables=tables, edges=edges, verbose=verbose)
    for t in ('customers', 'sellers'):
        g[t]['zip_prefix'].stype = 'categorical'
    return g.validate()


graph = make_graph(verbose=True)
graph['order_items'].metadata

# %% [markdown]
# ### 6c. Offline PQL check (optional)
#
# The NIM validates every query, but while you iterate it helps to catch typos without a GPU. The
# helper below calls the query validator bundled with the installed engine (version 1.0.x). This
# is an **internal** function, not the supported client API, so it is wrapped to fail softly if a
# future version moves it. Treat it as a linter. The NIM has the final word.

# %%
def check_pql(query: str, g=None) -> bool:
    g = g or graph
    try:
        from kumo_relational_engine.rfm.query_parser import parse_query_locally
        parse_query_locally(query, g._to_api_graph_definition())
    except ImportError:
        print('offline PQL check unavailable in this engine version'); return True
    except ValueError as error:
        print('✗', str(error).split('. ', 1)[-1][:300]); return False
    print('✓', query); return True


check_pql('PREDICT SUM(order_items.price, 0, 30, days) FOR EACH sellers.seller_id')
check_pql('PREDICT SUM(order_items.revenue, 0, 30, days) FOR EACH sellers.seller_id')        # no such column
check_pql("PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR customers.customer_id IN ('a')")     # 1-element list
check_pql('PREDICT LIST_DISTINCT(order_items.order_item_key, 0, 30, days) RANK TOP 5 FOR EACH sellers.seller_id')  # PK target

# Why section 4 put customer_id on order_items: without that direct link, use case 5 can't be asked.
# (Variant graphs on the same file must be built *before* the first predict(). See the README.)
no_shortcut = make_graph(edges=[e for e in EDGES if e != ('order_items', 'customer_id', 'customers')])
check_pql('PREDICT LIST_DISTINCT(order_items.product_id, 0, 90, days) RANK TOP 10 FOR EACH customers.customer_id', no_shortcut)
del no_shortcut

# %% [markdown]
# ## 7. Connect to the Kumo Relational NIM
#
# `RelationalClient` owns one connection to one NIM. No request is sent until `health_ready()` or
# the first prediction. `client.relational(graph)` returns a handle bound to the graph. The
# expensive part of a prediction, materializing the graph, is cached on that handle, so asking
# many questions of the same graph is cheap.

# %%
from kumo_relational_client import RelationalClient, RelationalError

client = RelationalClient(url=NIM_URL, api_key=NIM_API_KEY, timeout=300, max_retries=3)
try:
    NIM_READY = client.health_ready()
except RelationalError as error:
    NIM_READY = False
    print(f'NIM not reachable at {NIM_URL}: {error}')
print('NIM ready:', NIM_READY, '|', client, '|', client.capabilities('kumo-relational'))
model = client.relational(graph)


def kumo_predict(query: str, indices, g_model=None, **options):
    """model.predict with timing; returns None (and says so) when no NIM is up."""
    if not NIM_READY:
        print(f'⏭  skipped (no NIM): {query}')
        return None
    t0 = time.perf_counter()
    out = (g_model or model).predict(query, indices=list(indices), run_mode=RUN_MODE,
                                     batch_size='max', verbose=False, **options)
    print(f'✓ {len(indices):,} entities in {time.perf_counter() - t0:.1f}s ({RUN_MODE})')
    return out

# %% [markdown]
# ## 8. PQL in five minutes
#
# ```
# PREDICT <target>  FOR EACH <entity_table>.<primary_key>  [WHERE <entity filter>]  [ASSUMING <condition>]
# ```
#
# | Piece | Example | Meaning |
# |---|---|---|
# | Temporal aggregate | `SUM(order_items.price, 0, 30, days)` | Sum over the 30 days after the anchor time. The aggregated table must link directly to the entity |
# | Past window | `COUNT(orders.*, -90, 0, days)` | Negative offsets look back. Used in `WHERE` filters |
# | Comparison → classification | `COUNT(orders.*, 0, 90, days) > 0` | Any boolean expression makes a binary task |
# | Static column | `products.category` | Predict a column of the entity table itself (missing-value imputation) |
# | Filtered aggregate | `COUNT(reviews.* WHERE reviews.review_score <= 2, 0, 30, days)` | Aggregate only matching rows |
# | Ranking | `LIST_DISTINCT(order_items.product_id, 0, 90, days) RANK TOP 10` | Recommend FK targets (link prediction) |
# | Entities | `FOR EACH sellers.seller_id` plus `indices=[...]` | `indices` picks who gets scored (≤1000 per request; `batch_size='max'` splits for you) |
# | Entity filter | `WHERE COUNT(order_items.*, -90, 0, days) > 0` | Restricts which historical entities become **in-context examples**. Every id in `indices` is still scored |
#
# Aggregations available: `SUM AVG MIN MAX COUNT COUNT_DISTINCT FIRST LAST LIST_DISTINCT`. Time units: `days`, `hours`, `months`, ...
#
# **How it predicts without training.** For a query and an anchor time *t*, Kumo:
# 1. samples historical entities at earlier anchors (*t − 30d*, *t − 60d*, ...) and computes their
#    true labels from the data. These are the in-context examples. `run_mode` sets how many: 100,
#    1k, 5k or 10k.
# 2. samples each entity's multi-hop subgraph (its orders, their items, reviews, deliveries...)
#    using only rows older than that entity's anchor.
# 3. sends the labelled context subgraphs plus the unlabelled query subgraphs to the
#    pre-trained model. The model infers the pattern in one forward pass. There is no gradient step.

# %% [markdown]
# ### Backtesting setup
#
# To grade predictions you need the answer key, so we **predict as of a past date**. Kumo sees only
# rows older than `anchor_time`, and SQL computes what really happened afterwards. The classic
# baseline trains on monthly snapshots that all end before the test anchor.

# %%
TEST_ANCHOR = pd.Timestamp('2018-06-01')
TRAIN_ANCHORS = pd.date_range('2017-09-01', '2018-05-01', freq='MS')   # 30-day labels end <= 2018-05-31
EVAL_N = 1000                                                          # entities scored per use case
rng = np.random.default_rng(SEED)

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, mean_absolute_error,
                             roc_auc_score)
from scipy.stats import spearmanr

results = []   # (use case, method, metric, value) for the final scoreboard


def record(case, method, **metrics):
    for name, value in metrics.items():
        results.append({'use_case': case, 'method': method, 'metric': name, 'value': round(float(value), 4)})
    print(f'{case:<28} {method:<22}', '  '.join(f'{k}={v:.3f}' for k, v in metrics.items()))

# %% [markdown]
# ## 9. Use case 1: seller churn (binary classification)
#
# *Will an active seller make **no sales at all** in the next 30 days?* A marketplace team uses this
# to decide whom to call before a seller drifts away.
#
# ### The classic way: features in SQL, then train a model
#
# This is the work Kumo replaces. You pick windows, aggregate across three tables, guard
# against leakage at every join, stack monthly snapshots, then train. The query below is the
# feature store for this one task.

# %%
SELLER_SNAPSHOT_SQL = """
WITH hist AS (
    SELECT seller_id,
           SUM(CASE WHEN purchased_at >= datetime(:a, '-30 days') THEN price ELSE 0 END) AS rev_30d,
           SUM(CASE WHEN purchased_at >= datetime(:a, '-90 days') THEN price ELSE 0 END) AS rev_90d,
           SUM(price)                                                                    AS rev_all,
           SUM(purchased_at >= datetime(:a, '-30 days'))                                 AS items_30d,
           SUM(purchased_at >= datetime(:a, '-90 days'))                                 AS items_90d,
           COUNT(*)                                                                      AS items_all,
           COUNT(DISTINCT product_id)                                                    AS n_products,
           julianday(:a) - julianday(MAX(purchased_at))                                  AS days_since_sale,
           julianday(:a) - julianday(MIN(purchased_at))                                  AS tenure_days
    FROM order_items WHERE purchased_at < :a GROUP BY seller_id),
rev AS (
    SELECT oi.seller_id, AVG(r.review_score) AS avg_review, AVG(r.review_score <= 2) AS low_review_rate
    FROM reviews r JOIN order_items oi USING (order_id)
    WHERE r.created_at < :a GROUP BY oi.seller_id),
late AS (
    SELECT oi.seller_id, AVG(d.days_late > 0) AS late_rate, AVG(d.days_late) AS avg_days_late
    FROM deliveries d JOIN order_items oi USING (order_id)
    WHERE d.delivered_at < :a GROUP BY oi.seller_id),
future AS (
    SELECT seller_id, COUNT(*) AS items_next_30d, SUM(price) AS rev_next_30d
    FROM order_items WHERE purchased_at >= :a AND purchased_at < datetime(:a, '+30 days')
    GROUP BY seller_id)
SELECT :a AS anchor, h.*, rev.avg_review, rev.low_review_rate, late.late_rate, late.avg_days_late, s.state,
       COALESCE(f.items_next_30d, 0) = 0 AS churn_30d,
       COALESCE(f.rev_next_30d, 0)       AS rev_next_30d
FROM hist h JOIN sellers s USING (seller_id)
LEFT JOIN rev USING (seller_id) LEFT JOIN late USING (seller_id) LEFT JOIN future f USING (seller_id)
WHERE h.items_90d > 0                       -- "active" = sold something in the last 90 days
"""


def snapshots(query, anchors):
    return pd.concat([sql(query, a=a.strftime(TS)) for a in anchors], ignore_index=True)


seller_train = snapshots(SELLER_SNAPSHOT_SQL, TRAIN_ANCHORS)
seller_test = sql(SELLER_SNAPSHOT_SQL, a=TEST_ANCHOR.strftime(TS))
seller_test = seller_test.sample(min(EVAL_N, len(seller_test)), random_state=SEED).reset_index(drop=True)
print(f'train rows: {len(seller_train):,} over {len(TRAIN_ANCHORS)} anchors | test sellers: {len(seller_test):,}')
print(f'churn rate train {seller_train.churn_30d.mean():.1%} | test {seller_test.churn_30d.mean():.1%}')
seller_test.head(3)

# %%
SELLER_FEATURES = ['rev_30d', 'rev_90d', 'rev_all', 'items_30d', 'items_90d', 'items_all', 'n_products',
                   'days_since_sale', 'tenure_days', 'avg_review', 'low_review_rate', 'late_rate', 'avg_days_late', 'state']


def to_X(df, cols):
    X = df[cols].copy()
    for col in X.select_dtypes('object'):
        X[col] = X[col].astype('category')
    return X


cats = pd.api.types.union_categoricals([seller_train.state.astype('category'), seller_test.state.astype('category')]).categories
for df in (seller_train, seller_test):
    df['state'] = pd.Categorical(df.state, categories=cats)

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, categorical_features='from_dtype', random_state=SEED)
clf.fit(to_X(seller_train, SELLER_FEATURES), seller_train.churn_30d)
p_gbm = clf.predict_proba(to_X(seller_test, SELLER_FEATURES))[:, 1]

y = seller_test.churn_30d.astype(int)
record('1 seller churn', 'naive: days since sale', auc=roc_auc_score(y, seller_test.days_since_sale),
       ap=average_precision_score(y, seller_test.days_since_sale))
record('1 seller churn', 'SQL features + GBM', auc=roc_auc_score(y, p_gbm), ap=average_precision_score(y, p_gbm))

# %% [markdown]
# ### The Kumo way: one line of PQL
#
# There are no features, labels or training here. `anchor_time=TEST_ANCHOR` makes Kumo act as if
# today were 2018-06-01, so the evaluation is honest. The `WHERE` clause makes the in-context
# examples active sellers too, which matches our definition.
#
# Binary output has one row per entity with `PREDICTION`, `FALSE_PROB`, `TRUE_PROB` and
# `ANCHOR_TIMESTAMP`. Always join on `ENTITY`, never on row position.

# %%
Q_CHURN = ('PREDICT COUNT(order_items.*, 0, 30, days) = 0 '
           'FOR EACH sellers.seller_id '
           'WHERE COUNT(order_items.*, -90, 0, days) > 0')
check_pql(Q_CHURN)

kumo_churn = kumo_predict(Q_CHURN, seller_test.seller_id, anchor_time=TEST_ANCHOR)
if kumo_churn is not None:
    display(kumo_churn.head())
    scored = seller_test[['seller_id', 'churn_30d']].merge(
        kumo_churn.rename(columns={'ENTITY': 'seller_id'}), on='seller_id')
    record('1 seller churn', 'Kumo (no training)', auc=roc_auc_score(scored.churn_30d, scored.TRUE_PROB),
           ap=average_precision_score(scored.churn_30d, scored.TRUE_PROB))

# %% [markdown]
# ## 10. Use case 2: seller revenue next 30 days (regression)
#
# Same entities and same snapshot. Only the target changes. With the classic approach you would
# retrain, retune and pick a loss. With Kumo you change one word: `COUNT(...) = 0` becomes `SUM(price)`.

# %%
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, categorical_features='from_dtype', random_state=SEED)
reg.fit(to_X(seller_train, SELLER_FEATURES), np.log1p(seller_train.rev_next_30d))   # log: heavy-tailed target
r_gbm = np.expm1(reg.predict(to_X(seller_test, SELLER_FEATURES)))

y_rev = seller_test.rev_next_30d
record('2 seller revenue', 'naive: last 30 days', mae=mean_absolute_error(y_rev, seller_test.rev_30d),
       spearman=spearmanr(y_rev, seller_test.rev_30d).statistic)
record('2 seller revenue', 'SQL features + GBM', mae=mean_absolute_error(y_rev, r_gbm),
       spearman=spearmanr(y_rev, r_gbm).statistic)

Q_REVENUE = ('PREDICT SUM(order_items.price, 0, 30, days) '
             'FOR EACH sellers.seller_id '
             'WHERE COUNT(order_items.*, -90, 0, days) > 0')
check_pql(Q_REVENUE)
kumo_rev = kumo_predict(Q_REVENUE, seller_test.seller_id, anchor_time=TEST_ANCHOR)
if kumo_rev is not None:
    scored = seller_test[['seller_id', 'rev_next_30d']].merge(
        kumo_rev.rename(columns={'ENTITY': 'seller_id'}), on='seller_id')
    record('2 seller revenue', 'Kumo (no training)', mae=mean_absolute_error(scored.rev_next_30d, scored.PREDICTION),
           spearman=spearmanr(scored.rev_next_30d, scored.PREDICTION).statistic)

# %% [markdown]
# **Reading MAE on a heavy-tailed target:** a few big sellers dominate MAE. Spearman (rank
# correlation) tells you whether the model *orders* sellers correctly, which is usually what a
# prioritization list needs. For quantiles instead of a point estimate, pass
# `inference_config=dict(output_type='quantiles')`.

# %% [markdown]
# ## 11. Use case 3: will a customer buy again within 90 days? (rare positives)
#
# Olist is famous for having almost no repeat buyers (~3%). This is a hard and honest test.
# Ranking metrics (AUC, average precision) matter here, and accuracy is meaningless. The entity is
# `customers` and the target table is `orders`, one hop away.
#
# The horizon is 90 days, so the test anchor moves to 2018-05-01. The 90-day window then still ends
# inside the data.

# %%
CUST_ANCHOR = pd.Timestamp('2018-05-01')
CUST_TRAIN_ANCHORS = pd.date_range('2017-05-01', '2018-01-01', freq='MS')   # labels end <= 2018-04-01

CUSTOMER_SNAPSHOT_SQL = """
WITH hist AS (
    SELECT o.customer_id,
           COUNT(*)                                             AS n_orders,
           julianday(:a) - julianday(MAX(o.purchased_at))       AS days_since_order,
           julianday(:a) - julianday(MIN(o.purchased_at))       AS tenure_days,
           SUM(p.payment_value)                                 AS total_spend,
           MAX(p.installments)                                  AS max_installments
    FROM orders o LEFT JOIN payments p USING (order_id)
    WHERE o.purchased_at < :a GROUP BY o.customer_id),
rev AS (
    SELECT o.customer_id, AVG(r.review_score) AS avg_review, MIN(r.review_score) AS min_review
    FROM reviews r JOIN orders o USING (order_id)
    WHERE r.created_at < :a GROUP BY o.customer_id),
late AS (
    SELECT o.customer_id, MAX(d.days_late) AS max_days_late
    FROM deliveries d JOIN orders o USING (order_id)
    WHERE d.delivered_at < :a GROUP BY o.customer_id),
future AS (
    SELECT DISTINCT customer_id FROM orders
    WHERE purchased_at >= :a AND purchased_at < datetime(:a, '+90 days'))
SELECT :a AS anchor, h.*, rev.avg_review, rev.min_review, late.max_days_late, c.state,
       f.customer_id IS NOT NULL AS buys_again_90d
FROM hist h JOIN customers c USING (customer_id)
LEFT JOIN rev USING (customer_id) LEFT JOIN late USING (customer_id) LEFT JOIN future f USING (customer_id)
WHERE h.days_since_order <= 365
"""
cust_train = snapshots(CUSTOMER_SNAPSHOT_SQL, CUST_TRAIN_ANCHORS)
cust_test = sql(CUSTOMER_SNAPSHOT_SQL, a=CUST_ANCHOR.strftime(TS))
cust_test = cust_test.sample(min(EVAL_N, len(cust_test)), random_state=SEED).reset_index(drop=True)
print(f'train rows {len(cust_train):,} (pos {cust_train.buys_again_90d.mean():.2%}) | '
      f'test {len(cust_test):,} (pos {cust_test.buys_again_90d.mean():.2%})')

CUST_FEATURES = ['n_orders', 'days_since_order', 'tenure_days', 'total_spend', 'max_installments',
                 'avg_review', 'min_review', 'max_days_late', 'state']
cats = pd.api.types.union_categoricals([cust_train.state.astype('category'), cust_test.state.astype('category')]).categories
for df in (cust_train, cust_test):
    df['state'] = pd.Categorical(df.state, categories=cats)
clf_c = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, categorical_features='from_dtype', random_state=SEED)
clf_c.fit(to_X(cust_train, CUST_FEATURES), cust_train.buys_again_90d)
p_c = clf_c.predict_proba(to_X(cust_test, CUST_FEATURES))[:, 1]
y_c = cust_test.buys_again_90d.astype(int)
if y_c.nunique() == 2:
    record('3 customer repurchase', 'naive: recency', auc=roc_auc_score(y_c, -cust_test.days_since_order),
           ap=average_precision_score(y_c, -cust_test.days_since_order))
    record('3 customer repurchase', 'SQL features + GBM', auc=roc_auc_score(y_c, p_c), ap=average_precision_score(y_c, p_c))

Q_REPURCHASE = ('PREDICT COUNT(orders.*, 0, 90, days) > 0 '
                'FOR EACH customers.customer_id '
                'WHERE COUNT(orders.*, -365, 0, days) > 0')
check_pql(Q_REPURCHASE)
kumo_c = kumo_predict(Q_REPURCHASE, cust_test.customer_id, anchor_time=CUST_ANCHOR,
                      max_pq_iterations=50)   # rare positives -> let Kumo sample longer for labels
if kumo_c is not None and y_c.nunique() == 2:
    scored = cust_test[['customer_id', 'buys_again_90d']].merge(
        kumo_c.rename(columns={'ENTITY': 'customer_id'}), on='customer_id')
    record('3 customer repurchase', 'Kumo (no training)', auc=roc_auc_score(scored.buys_again_90d, scored.TRUE_PROB),
           ap=average_precision_score(scored.buys_again_90d, scored.TRUE_PROB))

# %% [markdown]
# ## 12. Use case 4: fill in missing product categories (multiclass on a static column)
#
# About 2% of Olist products have no category. Catalog teams fill these in by hand. Here the target
# is a **column of the entity table itself**, so there is no time window. Kumo learns from the
# products that do have a category and from each product's relational neighbourhood: who sells
# it, what else that seller sells, and what people paid.
#
# To grade it, we **hide** the category of 300 labelled products. We make one more Kumo-only
# snapshot, `olist_kumo_eval.db`, and blank the categories there before any graph opens it. The
# masked graph points at that file. Its tables, edges and query are identical. The baselines
# below apply the same mask in pandas.

# %%
known = sql("""SELECT p.product_id, p.category FROM products p
               WHERE p.category IS NOT NULL
                 AND EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.product_id)""")
holdout = known.sample(min(300, len(known)), random_state=SEED).reset_index(drop=True)

EVAL_DB_PATH = snapshot(DB_PATH, WORK / 'olist_kumo_eval.db')
with closing(sqlite3.connect(EVAL_DB_PATH)) as con:     # last time sqlite3 touches this file
    con.executemany('UPDATE products SET category = NULL WHERE product_id = ?', [(pid,) for pid in holdout.product_id])
    con.commit()
    n_null = con.execute('SELECT COUNT(*) FROM products WHERE category IS NULL').fetchone()[0]
print(f'held out: {len(holdout)} | NULL categories in {EVAL_DB_PATH.name}: {n_null} (holdout + originally missing)')

graph_masked = make_graph(db=EVAL_DB_PATH)

# %% [markdown]
# **Classic baselines.** Product attributes alone (weight, photos, description length) are weak.
# The relational signal is strong: "this seller's other products are mostly `housewares`". In
# classic ML you have to know to build that feature. Kumo finds it by walking
# `products → order_items → sellers → order_items → products`.

# %%
prod = sql("""
SELECT p.*, (SELECT oi.seller_id FROM order_items oi WHERE oi.product_id = p.product_id LIMIT 1) AS seller_id,
       (SELECT AVG(oi.price) FROM order_items oi WHERE oi.product_id = p.product_id) AS avg_price
FROM products p
""")
prod.loc[prod.product_id.isin(holdout.product_id), 'category'] = None      # same mask as olist_kumo_eval.db
# Relational feature: the seller's category mix over its *other*, still-labelled products.
seller_mix = (prod.dropna(subset=['category']).groupby(['seller_id', 'category']).size()
              .rename('n').reset_index())
top = seller_mix.sort_values('n', ascending=False).drop_duplicates('seller_id')
share = top.set_index('seller_id').n / seller_mix.groupby('seller_id').n.sum()
prod['seller_top_category'] = prod.seller_id.map(top.set_index('seller_id').category).astype('category')
prod['seller_top_share'] = prod.seller_id.map(share)

train_p = prod[prod.category.notna()]
test_p = holdout.merge(prod.drop(columns='category'), on='product_id')
BASE_COLS = ['name_length', 'description_length', 'photos_qty', 'weight_g', 'volume_cm3', 'avg_price']
for label, cols in [('product attrs + GBM', BASE_COLS),
                    ('+ seller-mix SQL feature', BASE_COLS + ['seller_top_category', 'seller_top_share'])]:
    m = HistGradientBoostingClassifier(max_iter=200, categorical_features='from_dtype', random_state=SEED)
    m.fit(train_p[cols], train_p.category)
    proba = m.predict_proba(test_p[cols])
    top3 = np.array(m.classes_)[np.argsort(-proba, axis=1)[:, :3]]
    record('4 product category', label, acc=accuracy_score(test_p.category, m.classes_[proba.argmax(1)]),
           top3=np.mean([t in row for t, row in zip(test_p.category, top3)]))

# %%
Q_CATEGORY = 'PREDICT products.category FOR EACH products.product_id'
check_pql(Q_CATEGORY, graph_masked)
kumo_cat = kumo_predict(Q_CATEGORY, holdout.product_id, g_model=client.relational(graph_masked))
if kumo_cat is not None:
    # Multiclass output: one row per (entity, class) with SCORE; PREDICTED marks the winner.
    ranked = kumo_cat.sort_values(['ENTITY', 'SCORE'], ascending=[True, False])
    top1 = ranked[ranked.PREDICTED].set_index('ENTITY').CLASS
    top3 = ranked.groupby('ENTITY').CLASS.apply(lambda s: list(s[:3]))
    truth = holdout.set_index('product_id').category
    record('4 product category', 'Kumo (no training)', acc=(top1.reindex(truth.index) == truth).mean(),
           top3=np.mean([t in top3.get(pid, []) for pid, t in truth.items()]))
    display(ranked.head(6))

# %% [markdown]
# ## 13. Use case 5: what will a customer buy next? (ranking)
#
# `LIST_DISTINCT(<fk column>, ...) RANK TOP k` turns Kumo into a recommender. It scores candidate
# rows of the table the FK points to (here `products`), and the output is *k* rows per entity
# (`CLASS`, `SCORE`).
#
# PQL aggregates a table that links **directly** to the entity. `order_items` hangs off `orders`, not
# `customers`, which is why section 4 copied `customer_id` onto `order_items` as a second FK. With
# it, the graph has a direct `order_items → customers` link. Denormalizing a key onto the fact
# table you predict over is the standard move. Section 6c showed the validator rejecting this
# query on a graph without that link.

# %%
Q_NEXT_PRODUCTS = ('PREDICT LIST_DISTINCT(order_items.product_id, 0, 90, days) RANK TOP 10 '
                   'FOR EACH customers.customer_id')
check_pql(Q_NEXT_PRODUCTS)

# Score customers who really did buy in the window, and measure hit@10 against a popularity baseline.
buyers = sql("""SELECT customer_id, GROUP_CONCAT(product_id) AS bought FROM order_items
                WHERE purchased_at >= :a AND purchased_at < datetime(:a, '+90 days')
                  AND customer_id IN (SELECT customer_id FROM orders WHERE purchased_at < :a)
                GROUP BY customer_id""", a=CUST_ANCHOR.strftime(TS))
buyers = buyers.sample(min(200, len(buyers)), random_state=SEED)          # 200 = per-request cap for ranking
truth_sets = {c: set(b.split(',')) for c, b in zip(buyers.customer_id, buyers.bought)}
popular = sql("""SELECT product_id FROM order_items WHERE purchased_at >= datetime(:a, '-90 days') AND purchased_at < :a
                 GROUP BY product_id ORDER BY COUNT(*) DESC LIMIT 10""", a=CUST_ANCHOR.strftime(TS)).product_id
record('5 next products', 'popular last 90 days', hit10=np.mean([bool(s & set(popular)) for s in truth_sets.values()]))

kumo_reco = kumo_predict(Q_NEXT_PRODUCTS, buyers.customer_id, anchor_time=CUST_ANCHOR)
if kumo_reco is not None:
    recs = kumo_reco.groupby('ENTITY').CLASS.apply(set)
    record('5 next products', 'Kumo (no training)', hit10=np.mean([bool(s & recs.get(c, set())) for c, s in truth_sets.items()]))

# %% [markdown]
# ## 14. Scoreboard
#
# One row per use case and method. With a NIM attached, the `Kumo (no training)` rows appear next to
# the baselines. Read them against the effort column of the README: the baselines took a
# hand-written SQL feature store, snapshot logic and a training run per task. Kumo took one PQL string.

# %%
board = pd.DataFrame(results)
display(board.pivot_table(index=['use_case', 'method'], columns='metric', values='value', sort=False))

primary = {'1 seller churn': 'auc', '2 seller revenue': 'spearman', '3 customer repurchase': 'auc',
           '4 product category': 'acc', '5 next products': 'hit10'}
view = board[board.apply(lambda r: primary.get(r.use_case) == r.metric, axis=1)]
fig, axes = plt.subplots(1, len(primary), figsize=(13, 2.9), sharey=False)
for ax, (case, metric) in zip(axes, primary.items()):
    part = view[view.use_case == case]
    colors = [SERIES[1] if m.startswith('Kumo') else SERIES[0] for m in part.method]
    ax.barh(part.method, part.value, color=colors, height=.6)
    for yy, v in enumerate(part.value):
        ax.text(v, yy, f' {v:.2f}', va='center', fontsize=8, color=INK)
    ax.set_title(f'{case}\n({metric})', fontsize=9)
    ax.set_xlim(0, max(1.0, part.value.max() * 1.25))
    ax.invert_yaxis(); ax.tick_params(labelsize=7); ax.grid(axis='y', visible=False)
fig.suptitle('Backtest scoreboard (blue = classic baselines, orange = Kumo)', fontsize=10, color=MUTED, x=.01, ha='left')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 15. Why did it predict that? Explanations
#
# `explain=...` returns an `Explanation` for a **single** entity. `explanation.prediction` holds the
# usual frame, and `details` holds structured attribution: which columns and which neighbouring rows
# drove the score.
#
# > **Data egress:** with the `[explain]` extra and the `KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY` set,
# > the client writes a natural-language summary by sending the entity's raw subgraph values to an
# > external LLM endpoint. `skip_summary=True` keeps everything local, and we use it here.

# %%
if NIM_READY:
    riskiest = (kumo_churn.sort_values('TRUE_PROB', ascending=False).ENTITY.iloc[0]
                if kumo_churn is not None else seller_test.seller_id.iloc[0])
    explanation = model.predict(Q_CHURN, indices=[riskiest], anchor_time=TEST_ANCHOR, run_mode='fast',
                                explain=dict(skip_summary=True), verbose=False)
    display(explanation.prediction)
    try:
        explanation.print()
    except Exception as error:      # printing helpers vary by engine version
        print(type(explanation.details), str(explanation.details)[:1500], error)
else:
    print('⏭  skipped (no NIM)')

# %% [markdown]
# ## 16. Bring your own labels: `predict_task` with SQL-built context
#
# Sometimes the label is not expressible in PQL. It might come from a CRM table, a manual review,
# or a business rule spread over five tables. Then **you** supply the in-context examples as a
# DataFrame (`ENTITY`, `ANCHOR_TIMESTAMP`, `TARGET`), and Kumo still pulls features from the graph.
#
# The seller snapshot we built for the baseline already *is* such a table.

# %%
CONTEXT_CAP = {'debug': 100, 'fast': 1_000, 'normal': 5_000, 'best': 10_000}[RUN_MODE]
context = (seller_train.sample(min(CONTEXT_CAP, len(seller_train)), random_state=SEED)
           .rename(columns={'seller_id': 'ENTITY', 'churn_30d': 'TARGET'})
           .assign(ANCHOR_TIMESTAMP=lambda d: pd.to_datetime(d.anchor), TARGET=lambda d: d.TARGET.astype(bool))
           [['ENTITY', 'ANCHOR_TIMESTAMP', 'TARGET']])
to_score = pd.DataFrame({'ENTITY': seller_test.seller_id, 'ANCHOR_TIMESTAMP': TEST_ANCHOR})
display(context.head(3))

if NIM_READY:
    byo = model.predict_task(context=context, predict=to_score, task_type='binary_classification',
                             entity_table='sellers', run_mode=RUN_MODE, verbose=False)
    scored = seller_test[['seller_id', 'churn_30d']].merge(byo.rename(columns={'ENTITY': 'seller_id'}), on='seller_id')
    record('1 seller churn', 'Kumo predict_task (SQL labels)', auc=roc_auc_score(scored.churn_30d, scored.TRUE_PROB),
           ap=average_precision_score(scored.churn_30d, scored.TRUE_PROB))
else:
    print('⏭  skipped (no NIM)')

# %% [markdown]
# ## 17. Production scoring: predict *now* and write back to SQLite
#
# For live use, drop `anchor_time`. Kumo then anchors at the latest timestamp in the data, which is
# "today". The scores go into a `predictions` table, so any app, dashboard or MCP tool that can read
# SQLite can use them with a join.
#
# Following the rule from section 4, predictions go to **their own file**, `predictions.db`. That
# file `ATTACH`es `olist.db` **read-only** for the join, so the store Kumo reads is never written by
# this process. In production this separation is also handy: the scoring job owns one file and the
# source data stays untouched. Without a NIM, the cell writes the classic model's scores
# instead, so you can still try the write-back and the join.

# %%
PRED_DB_PATH = WORK / 'predictions.db'
latest = sql('SELECT MAX(purchased_at) AS t FROM order_items').t[0]
now_snapshot = sql(SELLER_SNAPSHOT_SQL, a=latest)                    # features as of "today"
now_snapshot['state'] = pd.Categorical(now_snapshot.state, categories=seller_train.state.cat.categories)
print(f'{len(now_snapshot):,} active sellers to score as of {latest}')

live = kumo_predict(Q_CHURN, now_snapshot.seller_id)
if live is not None:
    scores = pd.DataFrame({'model': 'kumo-relational', 'entity_id': live.ENTITY.astype(str),
                           'score': live.TRUE_PROB,
                           'anchor_time': pd.to_datetime(live.ANCHOR_TIMESTAMP).dt.strftime(TS)})
else:
    scores = pd.DataFrame({'model': 'classic-gbm', 'entity_id': now_snapshot.seller_id,
                           'score': clf.predict_proba(to_X(now_snapshot, SELLER_FEATURES))[:, 1],
                           'anchor_time': latest})
scores = scores.assign(query=Q_CHURN, entity_table='sellers', scored_at=pd.Timestamp.now().strftime(TS))

con = sqlite3.connect(PRED_DB_PATH.as_uri(), uri=True)               # uri=True also enables URI ATTACH
try:
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
                     model TEXT, query TEXT, entity_table TEXT, entity_id TEXT,
                     score REAL, anchor_time TEXT, scored_at TEXT)""")
    scores.to_sql('predictions', con, if_exists='append', index=False)
    con.commit()
    con.execute(f"ATTACH DATABASE '{DB_PATH.as_uri()}?mode=ro' AS olist")
    worklist = pd.read_sql_query("""
        SELECT p.model, p.entity_id AS seller_id, s.state, ROUND(p.score, 3) AS churn_risk,
               ROUND(SUM(oi.price)) AS revenue_last_90d
        FROM predictions p
        JOIN olist.sellers s ON s.seller_id = p.entity_id
        JOIN olist.order_items oi ON oi.seller_id = p.entity_id
             AND oi.purchased_at >= datetime(p.anchor_time, '-90 days') AND oi.purchased_at < p.anchor_time
        WHERE p.query = :q AND p.scored_at = :t
        GROUP BY p.model, p.entity_id
        ORDER BY churn_risk * revenue_last_90d DESC LIMIT 10""", con,
        params={'q': Q_CHURN, 't': scores.scored_at.iloc[0]})
finally:
    con.close()
print('revenue at risk: churn probability x last-90-day revenue')
display(worklist)

# %% [markdown]
# ## 18. Clean up and next steps
#
# * `client.close()`, or use `with RelationalClient(...) as client:`, releases pooled connections.
# * Swap SQLite for production storage by changing one constructor: `Graph.from_duckdb`,
#   `from_snowflake`, `from_databricks`, `from_postgres`. PQL and `predict()` stay the same.
# * Try `run_mode='normal'` or `'best'` (5k / 10k in-context examples) on the use cases where
#   `fast` and the baseline are close.
# * Ask a new question: `PREDICT AVG(reviews.review_score, 0, 30, days) FOR EACH ...` needs only a
#   direct link from `reviews` to your entity. Add the FK in the schema build, as section 4 did for
#   `order_items.customer_id`.
#
# See [`README.md`](README.md) for the concepts, a PQL cheat-sheet, gotchas, and how to reproduce
# this with the local `structured-data-models` library.

# %%
client.close()
board.to_csv(WORK / 'scoreboard.csv', index=False)
print('scoreboard ->', WORK / 'scoreboard.csv')

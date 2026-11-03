# Museum guide: recognising the style and genre of a painting

Final project of the course **MSA-DATB09-03 · Neural Networks & Deep Learning Foundations**.

A museum wants a guide that recognises the **style** (Impressionism, Baroque, Cubism…) or the **genre**
(portrait, landscape, still life…) of a painting from a photo, on modest kiosk CPUs. The director's question:
*is a neural network worth it, or would a simpler classical model do the job?*

The project compares, on the same WikiArt images and the same split:

1. a classical baseline (logistic regression on pixels),
2. a feed-forward network written with PyTorch primitives (deliverable 1),
3. the same network with regularisation, after diagnosing its training (deliverable 2),
4. a ResNet18 pretrained on ImageNet and fine-tuned (deliverable 3),

and ends with a small Streamlit app that shows the three most likely styles and genres of a photo.

## Results

Test accuracy, measured once on the test set (15 % of the images, never used for any choice).
All the numbers come from the files of [`results/`](results).

| Model | Genres (11 classes) | Styles (27 classes) |
|---|---|---|
| Chance | 9.1 % | 3.7 % |
| Logistic regression on 64 × 64 pixels | 28.9 % | 11.5 % |
| Feed-forward network (4 × 4 pooling, 256-128) | 26.9 % | 11.5 % |
| Feed-forward network, regularised | 27.9 % | 9.7 % |
| **ResNet18 fine-tuned (layer4, lr 1e-4)** | **59.8 %** | **37.7 %** |
| ResNet18, correct answer among its top 3 | 83.0 % | 64.8 % |

**Styles grouped into 18 categories.** Many mistakes of the ResNet are between close movements
(Northern / Early Renaissance, Baroque / Rococo, the three Cubisms). The 27 styles are grouped into
18 artistically coherent categories (module [`style_bis.py`](src/museum_guide/style_bis.py)); the probabilities
of the 27 styles are summed by category, without retraining:

| Reading of the style ResNet | Chance | Top-1 | Top-3 |
|---|---|---|---|
| 27 styles | 3.7 % | 37.7 % | 64.8 % |
| 18 categories | 5.6 % | 47.7 % | 75.8 % |

**Cost on a kiosk** (one CPU thread, one image): 0.04 to 0.2 ms for the logistic regression and the
feed-forward network, and 20 to 70 ms for the ResNet18 (45 MB) depending on the laptop used for the measure.

**Answer to the director.** On raw pixels, the feed-forward network does not beat the logistic regression:
it learns the training images by heart, and regularisation removes this memorisation without improving the
accuracy. The pretrained ResNet18 is about twice (genres) to three times (styles) more accurate, and it still
answers well below the time a visitor can notice. A neural network is worth it, provided it is a pretrained
convolutional network. Its accuracy on styles remains limited, so the app shows the three most likely answers
and a word of caution rather than a single answer presented as certain (see
[`07_analysis_and_perspectives`](notebooks/07_analysis_and_perspectives.ipynb)).

## Data

The images come from the [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) dataset, through the
course script [`scripts/make_wikiart_subset.py`](scripts/make_wikiart_subset.py). All the classes are kept:

- **styles**: the 27 styles, 300 images per style (fewer for the three rarest: Action painting 98,
  Analytical Cubism 56, Synthetic Cubism 94), 7,448 images;
- **genres**: the 11 genres, 300 images per genre, 3,300 images;
- 128 × 128 pixels, at most 50 images per named painter, seed 0.

The split is the same for every model: 70 % training, 15 % validation, 15 % test, with the same class
proportions (seed 0, [`data.py`](src/museum_guide/data.py)).

> **The WikiArt images are never committed nor published.** The `data/` folder is ignored by Git; each member
> regenerates the images with the script. Only the description of each subset is kept, in
> [`subsets/`](subsets) (exact command, dataset revision, counts per class).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/lcaquelin/Neural_Networks.git
cd Neural_Networks
uv sync
```

`uv sync` creates the virtual environment `.venv`, installs the dependencies of `pyproject.toml` and installs
the shared code of `src/museum_guide` as a package, so that every notebook can write
`from museum_guide.data import prepare_subset`. In VS Code, choose the `.venv` kernel for the notebooks.

Then build the two subsets (a few GB are streamed; the first run also builds a small label index):

```bash
uv run python scripts/make_wikiart_subset.py --label style --classes Impressionism Realism Romanticism Expressionism Post_Impressionism Symbolism Art_Nouveau Baroque Abstract_Expressionism Northern_Renaissance Naive_Art_Primitivism Cubism Rococo Color_Field_Painting Pop_Art Early_Renaissance High_Renaissance Minimalism Mannerism_Late_Renaissance Ukiyo_e Fauvism Pointillism Contemporary_Realism New_Realism Synthetic_Cubism Analytical_Cubism Action_painting --per-class 300 --size 128 --out data/wikiart_style_all

uv run python scripts/make_wikiart_subset.py --label genre --classes Unknown_Genre portrait landscape genre_painting religious_painting abstract_painting cityscape sketch_and_study still_life nude_painting illustration --per-class 300 --size 128 --out data/wikiart_genre_all
```

Add `--dry-run` first to check the plan without downloading images.

## Notebooks

Run them in this order, from the `notebooks/` folder. Each pair `_01` / `_02` applies the same method to the
styles and to the genres.

| Notebook | Content |
|---|---|
| `01_explore_data` | Number of images per class, share of unknown artists, a few sample images |
| `02_load_data` | Loading, stratified split and standardisation (the code later moved to `data.py`) |
| `03_01_baseline_style`, `03_02_baseline_genre` | Logistic regression: the number every network must beat |
| `04_01_feed_forward_style`, `04_02_feed_forward_genre` | Deliverable 1: eight tries, one setting changed at a time, final network chosen on validation |
| `05_01_regularization_style`, `05_02_regularization_genre` | Deliverable 2: over-fitting diagnosis, weight decay, dropout, batch norm, early stopping |
| `05_OLD_regularization` | First regularisation experiments, replaced by `05_01` and `05_02` |
| `06_01_resnet_style`, `06_02_resnet_genre` | Deliverable 3: ResNet18 (head, then layer4), CPU cost, comparison of all the models, top-3 output, 18 style categories, saving of the model for the app |
| `07_analysis_and_perspectives` | Why the accuracy is limited on styles, and ways to improve |

Every final model writes a results file in `results/` (`test_accuracy`, `cpu_ms_per_image`, `n_parameters`,
`size_mb`, `n_classes`…) and its figures in `results/figures/`; the comparison cells read these files.
The trained weights are saved in `data/models/` and stay out of Git.

## The museum guide app

[`streamlit_app.py`](streamlit_app.py) lets a visitor import a photo of a painting. It shows the three most
likely style categories with their descriptions (and the styles each one brings together), the three most likely
genres, and a word of caution: the analysis is generated by AI and may contain errors.

1. Run `06_01_resnet_style` and `06_02_resnet_genre` up to their last cell, which saves
   `data/models/resnet18_style_app.pt` and `data/models/resnet18_genre_app.pt`.
2. Start the app from the project root:

   ```bash
   uv run streamlit run streamlit_app.py
   ```

The "Take a photo" button is a placeholder and is not active yet.

## Repository structure

```
├── notebooks/                 # the experiments, in order (see above)
├── results/                   # one JSON file per final model
│   └── figures/               # learning curves, confusion matrices, comparisons
├── scripts/
│   └── make_wikiart_subset.py # course script that builds the WikiArt subsets
├── src/museum_guide/          # shared code, installed by uv sync
│   ├── data.py                # loading, stratified split, standardisation, DataLoaders
│   ├── training.py            # training loop, evaluation, parameter count
│   ├── style_bis.py           # the 18 style categories and their descriptions
│   └── predict.py             # predictions of the app (top-3 categories and genres of a photo)
├── subsets/                   # description of each subset (command, revision, counts)
├── streamlit_app.py           # the museum guide app
├── pyproject.toml, uv.lock    # dependencies (uv)
└── data/                      # images and model weights, ignored by Git
```

## Team workflow

Each piece of work starts with a GitHub Issue, is done on its own branch with small commits, and is merged
into `main` through a pull request that closes the Issue. The `main` branch is protected.

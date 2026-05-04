# 🎬 Movie Recommendation System

A robust Collaborative Filtering recommendation engine built from scratch using Python, Pandas data pipelines, and NumPy linear algebra. 

## ✨ Project Highlights
- **Set up a professional workspace:** Virtual environments, `.gitignore`, and modular folder structures (`src/`, `data/`).
- **Built a robust Data Pipeline:** `data_loader.py` dynamically grabs raw CSV data from disk and merges the disparate tables.
- **Data Reshaping:** `preprocessor.py` successfully pivoted a long receipt of 100,000 text ratings into a massive mathematical grid (sparse matrix).
- **Calculated Cosine Similarity:** `similarity.py` utilized raw NumPy arrays and vectorization to instantaneously execute parallel calculus and compare 610 users.
- **Built the Recommender Logic:** `recommender.py` parsed those mathematical output scores to cross-reference un-watched movies and return highly accurate top 5 results for any user.

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **Pandas**: Used for data ingestion, DataFrame merging, and creating sparse User-Item matrices.
- **NumPy**: Used for high-performance mathematical operations (Vectorization, Dot Products, Magnitudes).

## 📂 Project Structure
```text
movie-recommend/
│
├── data/                    # Ignored by git
│   ├── raw/                 # Original data (movies.csv, ratings.csv)
│   └── processed/           # Saved similarity matrices (.npy)
│
├── src/                     # Core pipeline modules
│   ├── data_loader.py       # Loads and merges datasets
│   ├── preprocessor.py      # Pivots data into User-Item matrix
│   ├── similarity.py        # Calculates Cosine Similarity math
│   └── recommender.py       # Recommendation logic & nearest neighbors
│
├── app.py                   # Main pipeline controller
├── requirements.txt         # Project dependencies
└── .gitignore               # Git ignore rules
```

## ⚙️ Data Pipeline Flow
The project relies on a sequential data pipeline executed in `app.py`:

1. **Ingestion (`src/data_loader.py`)**: 
   - Dynamically locates the `data/raw/` directory using `pathlib`.
   - Loads `movies.csv` and `ratings.csv` into memory.
   - Merges the two tables on `movieId` so ratings are linked to actual movie titles.
2. **Preprocessing (`src/preprocessor.py`)**: 
   - Takes the long "receipt-style" data and `pivot_table`s it into a massive 2D User-Item Matrix.
   - Rows = Users, Columns = Movies, Values = Ratings.
   - Fills missing values with `0` (Sparse Matrix).
3. **Mathematical Engine (`src/similarity.py`)**: 
   - Strips the Pandas labels to create a pure NumPy array for high-speed calculation.
   - Computes **Cosine Similarity** between all users simultaneously.
   - Automatically saves the resulting similarity matrix to `data/processed/`.
4. **Recommender Logic (`src/recommender.py`)**:
   - Takes a `target_user_id`.
   - Looks up the user's nearest mathematical neighbor in the similarity matrix.
   - Filters out movies the target user has already seen.
   - Cross-references the neighbor's ratings on those unseen movies and returns the top 5 highest-rated picks.

## 🚀 How to Run the Project

### 1. Setup Virtual Environment
Open your terminal in the project root folder and run:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download the Dataset
1. Visit the [MovieLens Datasets](https://grouplens.org/datasets/movielens/) page.
2. Download the **"ml-latest-small.zip"** dataset.
3. Extract the zip and place `movies.csv` and `ratings.csv` into the `data/raw/` directory.

### 4. Run the Pipeline
Once the environment is active and data is in place, execute the main script:
```powershell
python app.py
```
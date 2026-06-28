# 🐦 Twitter Sentiment Analysis

A data analysis project that performs sentiment analysis on tweets related to various brands/entities using Python and TextBlob.

---

## 📂 Dataset

**File:** `twitter_training.csv`  
**Total Records:** ~74,682 tweets  
**Columns:**

| Column | Description |
|---|---|
| `ID` | Unique tweet ID |
| `Entity` | Brand/game/product mentioned |
| `Sentiments` | Sentiment label (Positive, Negative, Neutral, Irrelevant) |
| `Content` | Tweet text |

**Entities covered:** Borderlands, Amazon, CallOfDutyBlackopsColdWar, Overwatch, Xbox, NBA2K, Dota2, PlayStation5, CS-GO, and more.

---

## 🛠️ Technologies Used

- **Python 3**
- **Pandas** – data loading, cleaning, exploration
- **TextBlob** – NLP / sentiment processing
- **Matplotlib** – data visualization

---

## 📊 Features

- Load and preview the dataset
- Data cleaning: remove null values and duplicates
- Sentiment distribution visualization (bar chart)
- Brand-specific sentiment analysis (e.g., Microsoft tweets)
- Pie chart for per-brand sentiment breakdown

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd twitter-sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas matplotlib textblob
   python -m textblob.download_corpora
   ```

3. **Place the dataset**  
   Make sure `twitter_training.csv` is in the same directory as the notebook.

4. **Run the notebook**
   ```bash
   jupyter notebook DS Task_04.ipynb
   ```

---

## 📁 Project Structure

```
twitter-sentiment-analysis/
│
├── twitter_training.csv       
├── Untitled__1_.ipynb        
└── README.md                 

---

## 📈 Analysis Steps

1. **Data Loading** – Read CSV with custom column names
2. **Exploration** – `.head()`, `.shape()`, `.describe()`
3. **Cleaning** – Drop nulls and duplicate rows
4. **Sentiment Distribution** – Bar chart of all 4 sentiment classes
5. **Brand Filtering** – Filter tweets by entity (e.g., Microsoft)
6. **Brand Sentiment Pie Chart** – Visual breakdown per brand

---

## 🙋‍♀️ Author

**Siddhi Shinde**  

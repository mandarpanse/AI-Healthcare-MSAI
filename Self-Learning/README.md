
# Healthcare Data Analysis - Project Guide

This project performs a comprehensive analysis on healthcare data, including data cleaning, exploratory data analysis (EDA), trend analysis, and visualization.

## Project Structure

- `Healthcare_Data_Analysis_Complete_With_Analysis.ipynb`: Jupyter Notebook containing the entire analysis process.
- `Healthcare_Data_Analysis_Presentation.pptx`: Final PowerPoint presentation with step-by-step instructions.
- `README.md`: Setup guide and usage instructions.

---

## Setup Instructions

### 1. **Clone Repository or Download Files**

Download all the project files, including:
- Jupyter Notebook
- Dataset (`healthcare_dataset.csv`)
- Presentation PPTX

---

### 2. **Environment Setup**

Ensure you have Python 3.7+ installed.

Create a virtual environment (optional but recommended):

```bash
python -m venv healthcare-env
source healthcare-env/bin/activate  # On Windows: healthcare-env\Scripts\activate
```

---

### 3. **Install Required Libraries**

Use `pip` to install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

For PowerPoint automation and exporting:

```bash
pip install python-pptx nbconvert
```

---

### 4. **Dataset Preparation**

Place the dataset CSV file (`healthcare_dataset.csv`) in the same directory as the notebook or adjust the path in the notebook:

```python
df = pd.read_csv('healthcare_dataset.csv')
```

---

### 5. **Running the Notebook**

Open the notebook in Jupyter:

```bash
jupyter notebook Healthcare_Data_Analysis_Complete_With_Analysis.ipynb
```

Run each cell sequentially to:

1. Import libraries.
2. Load & clean the data.
3. Perform EDA (visualizations included).
4. Conduct trend analysis.
5. View results.

---

## Key Features

- Data Cleaning: Standardization, date conversion, duplicate removal.
- EDA: Age distribution, gender ratio, billing analysis, medical condition frequency.
- Trend Analysis: Monthly admissions over time.
- Visualization: Matplotlib and Seaborn plots.
- Presentation: Detailed slides with speaker notes and instructions.

---

## Author

Mandar Panse | March 2025

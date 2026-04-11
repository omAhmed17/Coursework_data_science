# Program

Project directory structure:
- `data/raw/`: Original dataset (never touch this)
- `data/processed/`: Cleaned dataset (saved here after cleaning)
- `notebooks/`: 
  - `01_data_preparation.ipynb`: Cleans data & saves to data/processed/
  - `02_modeling.ipynb`: Loads data from data/processed/ & trains model
- `src/`: Backend / extra source code
  - `server.js`

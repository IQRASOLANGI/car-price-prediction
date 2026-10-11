# Car Price Prediction

A supervised machine learning project that estimates the selling price of a car from its age, mileage, fuel type, transmission, seller type, brand and ownership history. Three regression models are trained and compared on real CarDekho listings; the best, Gradient Boosting, reaches an **R² of 0.78** on held-out data.

## Results

Evaluated on a held-out test set of 1,343 cars (80/20 split, `random_state=42`). Prices are in Indian rupees, as in the source data.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Gradient Boosting** | **130,097** | **228,592** | **0.784** |
| Random Forest | 138,727 | 252,137 | 0.737 |
| Linear Regression | 171,820 | 332,886 | 0.541 |

The tree-based ensembles clearly outperform the linear baseline because price depends non-linearly on age, mileage and brand. For context, the median price in the cleaned data is ₹420,000, so the best model's average error is roughly ₹130k.

## Dataset

- **Source:** [Vehicle dataset from CarDekho](https://www.kaggle.com/datasets) (Kaggle). The CSV is not included in this repository.
- **Raw size:** 8,128 listings, 12 columns.
- **After cleaning:** 6,715 listings across 31 brands.
- **Target:** `selling_price`

## Approach

1. **Cleaning:** standardised column names, removed 1,202 duplicate rows and rows with missing values, normalised inconsistent text (e.g. `Petrol` vs `petrol`), mapped `owner` to an ordinal number, and dropped impossible odometer readings (over 500,000 km).
2. **Feature engineering:** `car_age` (current year minus manufacturing year) and `brand` (extracted from the car name).
3. **Exploratory analysis:** price distribution (raw and log scale), price by fuel type, price against car age, median price by brand, and a correlation heatmap.
4. **Preprocessing:** one-hot encoding of `brand`, `fuel`, `seller_type` and `transmission` inside a scikit-learn `Pipeline`, so the encoder is fitted on training data only and there is no leakage. Brands with fewer than 10 cars are grouped as "infrequent".
5. **Modelling:** Linear Regression, Random Forest (300 trees) and Gradient Boosting (300 trees), each trained on `log(price)` and converted back to rupees for evaluation. The log transform handles the strongly right-skewed price distribution.
6. **Evaluation:** MAE, RMSE and R² on the test set, plus feature importance and a worked example prediction.

## Key findings

- Price falls steadily with car age, which is one of the strongest drivers of resale value.
- Diesel cars tend to sell for more than petrol cars; CNG and LPG cars sit at the lower end.
- Car age, kilometres driven, fuel type and transmission carry most of the predictive weight, with some premium brands adding further uplift.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Repository structure

```
car-price-prediction/
├── README.md
├── car_price_prediction.ipynb
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/IQRASOLANGI/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

1. Download the CarDekho vehicle dataset from Kaggle and place the CSV in the project folder. The notebook picks up the first CSV whose filename contains `car` or `vehicle` (for example `cardekho.csv`).
2. Run `jupyter notebook` and open `car_price_prediction.ipynb`.
3. Choose **Run → Run All Cells**.

## Limitations

- The data has no information on vehicle condition, accident history, location, service history or trim level, all of which affect real prices.
- Performance is measured on a single train/test split; cross-validation would give a more reliable estimate.
- Predictions are estimates, not guaranteed market values.
- Different versions of the CarDekho dataset use different column names and price units, so check the file before running.

## Future improvements

- Hyperparameter tuning with `GridSearchCV` and k-fold cross-validation.
- Use the engine, power, mileage and seats columns that are currently left out.
- Deploy the model behind a simple web app for interactive price estimates.

## Author

**Iqra Solangi**
[GitHub](https://github.com/IQRASOLANGI) · [LinkedIn](https://www.linkedin.com/in/iqra-solangi-a65945334)

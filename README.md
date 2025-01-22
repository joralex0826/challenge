# MELI Data Scientist challenge

## Deliverables

This repository includes the following deliverables:

1. [Model Implementation](#1-model-implementation):  
   The file containing all the necessary code to define, train, and evaluate the model.

2. [Feature Selection and Performance Evaluation](#2-feature-selection-and-performance-evaluation):  
   A document explaining the criteria used for selecting features, the secondary evaluation metric applied, and the performance achieved on that metric.

3. [Exploratory Data Analysis (Optional)](#3-exploratory-data-analysis-optional):  
   Optionally, an EDA analysis in an alternative format (e.g., Jupyter Notebook `.ipynb`) can be included.

4. [Conclusions](#4-conclusions):  
   Conclusions and next steps.


### 1. Model Implementation

In the following module **[main.py](main.py)**, you will be able to execute the modeling and evaluation process for the XGBoost - BERT embeddings model. To take a closer look at the exploration with other models, you can refer to the **[notebook](notebooks/MELI_challenge.ipynb)**.

When running the process, it will train the model, save the `.pkl` file, and additionally, a summary of the obtained metrics will be printed on the screen.


In `requirements.txt` file, you will find the list of packages required to compile the process. If they are not installed already, you can execute:

`pip install -r requirements.txt`

To run the process correctly, it is necessary to have the `.parquet` and  `MLA_100k_checked_v3` files attached in the email with the solution. These files should be placed in the `notebooks` folder.


### 2. Feature Selection and Performance Evaluation

In this section, we provide an explanation of the criteria used to select the features, the secondary evaluation metric applied, and the performance achieved on that metric. 

In general, the methodology applied to select the features was based both on an analysis of the nature of the variable and its impact on the addressed problem, as well as on the relevant statistical analyses. Below, all the variables will be listed, along with a brief explanation of the treatment applied and whether or not they were chosen for the model.

#### Feature Selection Criteria

- **seller_address**: Variable removed in its original form; however, a new variable, "city," was created and used in the model.  
- **warranty**: Used in the model; its values were strategically grouped into four categories: with warranty, without warranty, others with warranty, and no information.  
- **sub_status**: Not considered in the model due to a high imbalance in the data (99% grouped into a single label).  
- **deal_ids**: Similar to sub_status, not considered in the model.  
- **base_price**: Used in the model, as it could be a useful variable (normalized during modeling).  
- **shipping**: Used in the model, but not in its original version; it was transformed into two new categorical variables: `local_pick_up` and `free_shipping`. The shipping method might provide information about whether a product is new or used.  
- **non_mercado_pago_payment_methods**: Used and transformed into new boolean variables. Thus, one variable was created for each payment method.  
- **seller_id**: Not considered, as it could lead to overfitting and prevent the model from learning general patterns.  
- **variations**: Not considered. This variable contained many product characteristics, similar to attributes. This option could be considered in the future, but for now, it does not seem very useful.  
- **site_id**: Not considered, as it contained a single unique value across all records.  
- **listing_type_id**: Used in the model; it could contain relevant information (a possible improvement would be to group less frequent data into a single variable).  
- **price**: Since it generally contains the same information as `base_price`, it was not retained.  
- **attributes**: Not considered. Although it contained information about the product, the variety of characteristics might lead the model to learn specific patterns rather than general ones.  
- **buying_mode**: Not considered due to imbalance.  
- **tags**: Used in the model; transformed into dummy variables, one variable per tag type.  
- **listing_source**: Removed, as all its values were null.  
- **parent_item_id**: Removed, as it does not provide valuable information for the model.  
- **coverage_areas**: Removed, as all its values were null.  
- **category_id**: Not considered in the model. However, as a future opportunity, `category_name` could be acquired to generate a new feature, even using embeddings.  
- **descriptions**: Removed, as it was a variable that contained only IDs.  
- **last_updated**: Used to create a new variable for the model: whether the model has been updated since creation.  
- **international_delivery_mode**: Removed, as it only contained null values.  
- **pictures**: Did not provide valuable information for the model.  
- **id**: Did not provide valuable information for the model.  
- **official_store_id**: Did not provide valuable information for the model.  
- **differential_pricing**: All values were null.  
- **accepts_mercadopago**: Not considered; the variable was highly imbalanced.  
- **original_price**: Similar to `base_price` but with more null values; thus, it was not considered.  
- **currency_id**: Not considered; the variable was highly imbalanced.  
- **thumbnail**: Removed, as it did not provide relevant information.  
- **title**: Used and transformed with embeddings; this variable could provide important information about the product's condition.  
- **automatic_relist**: Considered in the model, as it might include important information.  
- **date_created**: Not used; `start_time` was used instead.  
- **secure_thumbnail**: Removed, as it did not provide relevant information.  
- **stop_time**: Not used; another time variable was chosen instead.  
- **status**: Discarded, as it might not add value for future predictions. For example, all products entering the catalog would have the label `active`, which does not necessarily indicate whether the product is new or used.  
- **video_id**: Removed, as it did not provide relevant information.  
- **catalog_product_id**: Not considered; not relevant.  
- **subtitle**: Not considered; it only contained null values.  
- **initial_quantity**: Considered, as it could provide valuable information.  
- **start_time**: Considered and combined with `last_updated` to create a new variable.  
- **permalink**: Not used, as it does not provide relevant information.  
- **sold_quantity**: Considered, as it could provide information about the product's condition.  
- **available_quantity**: Removed, as it was highly correlated with `initial_quantity`.  

#### Features used by the model and it's transformations
- **base_price**: Normalized.
- **title**: Vectorized with BERT embeddings.
- **listing_type_id**: Dummies. 
- **buying_mode**: Group labels and Dummies. 
- **warranty**: Group labels and Dummies. 
- **tags**: Get clean labels and Dummies.
- **initial_quantity**: Normalized.
- **sold_quantity**: Normalized.
- **start_time**: Combined with `last_updated` to create two new variables: `updated_label` (if the label has been updated or not) and `updated_since_creation` (days after last updated).  
- **last_updated**: See `start_time`
- **shipping**: Group labels and Dummies. 
- **seller_address**: Get `city_name`. The frequency of each city were calculated, after that, this field were normalized.
- **non_mercado_pago_payment_methods**: Group labels and Dummies.
- **automatic_relist**: Transformed to boolean.

#### Secondary Metric

The secondary evaluation metric chosen for this analysis was f1-score, as it provides deeper insight into the precision and recall of the model. Unlike accuracy, which can be misleading in imbalanced datasets, the F1-score balances the trade-off between precision (the accuracy of positive predictions) and recall (the ability to find all relevant positive instances). This metric is especially useful when both false positives and false negatives have significant consequences, allowing for a better understanding of the model's performance.

#### Model Performance

The performance of the model on the metrics is summarized in the following table:

| Model Name         | Accuracy | F1-Score | Precision | Recall |
|--------------------|----------|----------|-----------|--------|
| Logistic regression - BERT embeddings | 0.84 | 0.84 | 0.84  | 0.83 |
| XGboost base | 0.83 | 0.83 | 0.83  | 0.83 |
| XGboost - BERT embeddings | 0.87 | 0.87 | 0.87  | 0.87 |
| Neural Network - BERT embeddings  | 0.84 | 0.84 | 0.84  | 0.84 |

Some conclusions:

- When precision and recall are equal or very similar for both classes, their F1-score (which is the harmonic mean of both) will also be equal. This shows that the model is not biased towards precision (avoiding false positives) or recall (avoiding false negatives), which is ideal in this scenario.

- The fact that the metrics are consistent and high (~0.87, in the best model) across all categories shows that the model is performing well and uniformly for both classes. It is not ignoring or prioritizing one class over the other, suggesting that it is robust.

- **Logistic Regression - BERT embeddings** can be considered an excellent option as it takes 1/4 of the time required to train the XGBoost (with better results). Additionally, its metrics are not far from the best.

- **XGBoost** is a good choice as a base model, as it is modeled using only 29 variables. Under this condition, different combinations can be tested, and performance can be measured.

- Although the **XGBoost - BERT embeddings** model presents the best results, its computational cost tends to be a bit higher, requiring a more powerful machine. However, the improvement in metrics compared to its base version is clear. The embeddings make the difference.

- For its complexity and low interpretability, I would recommend other models before **Neural Network - BERT embeddings**. However, as more data becomes available for training, this could become a better option.


### 3. Exploratory Data Analysis (Optional)

An optional Exploratory Data Analysis (EDA) is provided in a separate Jupyter Notebook. You can find the notebook at the following location:

- **[EDA Notebook](notebooks/MELI_challenge.ipynb)**

This notebook includes the entire process carried out, from EDA to modeling.


### 4. Conclusions

- **Model Selection:** It is advisable to use either the Logistic Regression - BERT embeddings model or the XGBoost - BERT embeddings model, depending on the available computational resources. Both options demonstrated good results and performance, with XGBoost meeting the minimum accuracy required in the challenge.

- **Continuous Training and Dataset Enrichment:** Regular training, along with enriching the training dataset with more data, could significantly improve the model's performance over time. Additionally, methods to obtain probabilities (e.g., using `model.predict_proba(data)`) could be applied to establish thresholds and estimate a "confidence level" for predictions. This approach would allow adding new data to the training dataset with higher certainty of proper classification.

- **Hyperparameter Tuning:** Due to computational limitations, further testing with different hyperparameters was not feasible. This could be better explored using grid search or Bayesian search, as well as experimenting with various combinations of variables.

- It is expected that the model, which currently has a good level of accuracy, will be improved over time and better results will be achieved.
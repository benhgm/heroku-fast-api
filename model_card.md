# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random forest classifier using sklearn.

## Intended Use

This model is used to predict if a particular person's income will exceed $50,000 per year based on census data

## Training / Evaluation Data

This model is trained using census data from the UCI Machine Learning Repository
Link to the dataset is found [here](https://archive.ics.uci.edu/ml/datasets/census+income)

The model is trained and validated using an 80-20 train-test split of the dataset.

## Metrics

The model is evaluated on the following metrics:
- Precision: 0.75
- Recall: 0.62
- Fbeta: 0.68

## Ethical Considerations

The data contains information related to sex, gender, education level and race. It must be noted that the results of the model should only be representative of the scope of the data and not to be taken out of context as a representation of the entire population.

## Caveats and Recommendations

The model performance was obtained using default hyperparameters of the sklearn Random Forest Classifier. Further improvements can be made to the model via hyperparameter tuning and expansion of the dataset.

# Customer Churn

Churn = The phenomena where a customer leaves an organization.

## Goal

* To create a model to predict whether or not a customer is likely to leave the bank based on various customer characteristics.
* Use **PyTorch**, which is a commonly used deep learning library developed by ***Facebook***, for the classification.

## Dependencies

* PyTorch
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
<br>

`pip install -r requirements.txt`

## Dataset

Kaggle Link: https://www.kaggle.com/kmalit/bank-customer-churn-prediction/data.

Saved in location: *data/bank_churn_data.csv*

Note: The values for the charcateristics are recorded 6 months before the value for the churn (exited or not) was obtained since the task is to predict customer churn after 6 months from the time when the customer information is recorded.

## Data Analysis

Notebook: data_analysis.ipynb

* *Columns*: 'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'
* **Unuseful Features**: 'RowNumber', 'CustomerId', 'Surname'
* **Categorical Features**: 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember'
* **Numerical Features**: 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'
* **Labels**: 'Exited'
* **Churn Ratio**: 20% of the customers left the bank.
* **Geographical distribution**: Almost 50% of customers are from France, while those from Spain and Germany are 25% each.
* **Geographical distribution of churn**
  * The same number for French and German customers left the bank. However, overall there are twice as many French customer as compare to German.
  * The number of German customers who left the bank is twice that of the Spanish customers. But, the overall number of German and Spanish customers is the same.
  * **Therefore**, German customers are more likely to leave the bank.

## Data Preprocessing

* Segregate the features and labels
* Convert the *type* for categorical feature-columns to '*category*'. 
  * When data type is changed to '*category*', each category in the feature-column is assigned a unique code.
* Convert categorical feature-columns to tensors.
  * Convert the categorical columns to their code-values array.
  * Convert the array to a tensor. 
  * Apply *torch.int64* as the data type for categorical feature-columns.
* Convert the numerical feature-columns to tensor.
  * Convert the numerical columns to their values array.
  * Convert the array to a tensor.
  * Apply *torch.float* as the data type for numerical feature-columns.
* Convert the label-column to tensor.
  * Obtain the label-column values
  * Convert them to a tensor
  * Flatten the same.

## Split data into training and test sets

* Training data = 80% of total records
* Test data = 20% of total records

## Create the model

### Prepare size for the *Embedding* layer

* Divide the no. of unique values in the column by 2 but <= 50

### Define the model

* The Model class inherits from ***PyTorch***'s **nn** module's ***Module*** class.
* Initializing:
  * Required Parameters
    * **embedding_size**: contains the list of embedding size for the categorical columns
    * **num_cols**: total number of numerical columns
    * **output_size**: number of possible outputs
    * **layers**: list of no. of neurons for all the layers
    * **p**: Dropout (default=0.5)
  * Initialized variables
    * **all_embeddings**: list of ***ModuleList*** objects for the embedding sizes of the categorical columns
    * **embedding_dropout**: ***Dropout*** value for all the layers
    * **batch_norm_num**: list of ***BatchNorm1d*** objects for all the numerical columns
    * **input_size**: size of the input layer; total number of numerical and categorical layers
  * Model layers
    * ***Linear***: to calculate dot product of input layer and weight matrices
    * ***ReLU***: applied as the activation function
    * ***BatchNorm1d***: to apply batch normalization to the numerical columns
    * ***Dropout***: to avoid overfitting
    * ***Linear***: for the output layer
  * All the layers set to exectute sequentially using ***Sequential*** class
* Passing the embedded categorical and numberical features as inputs to the model
  * Obtain the embedded categorical columns
  * Add the embedded categorical columns to embedded layer
  * Dropout to prevent overfitting
  * Normalize the embedded numerical columns
  * Add the embedded numerical columns to embedded layer
  * Pass the embedded layer to the sequential layers

### Set the model  

Pass the required parameter values:

* embedding size of the categorical columns
* the number of numerical columns
* the number of outputs (2, for exited or not exited)
* the list of neurons for the hidden layers (3: 200, 100, 50)
* the dropout value (0.4, default=0.5)

#### Observations

```
Model(
  (all_embeddings): ModuleList(
    (0): Embedding(3, 2)
    (1): Embedding(2, 1)
    (2): Embedding(2, 1)
    (3): Embedding(2, 1)
  )
  (embedding_dropout): Dropout(p=0.4, inplace=False)
  (batch_norm_num): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=11, out_features=200, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.4, inplace=False)
    (4): Linear(in_features=200, out_features=100, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.4, inplace=False)
    (8): Linear(in_features=100, out_features=50, bias=True)
    (9): ReLU(inplace=True)
    (10): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout(p=0.4, inplace=False)
    (12): Linear(in_features=50, out_features=2, bias=True)
  )
)
```

* In the first linear layer the value of the in_features variable is 11
  * 6 numerical columns + sum of embedding dimensions for the categorical columns 3+2 = 5
* In the last layer, the out_features has a value of 2 since we have only 2 possible outputs 1 or 0

## Train the model

* Define the loss function
* Define the optimizer function
* Set the number of epochs
* Iterate till the number of epochs to train the model
  * Pass the embedded training data to the model and predict the label
  * Calculate the loss using the loss function
  * Save the loss to the aggregated loss list
  * Set gradient to zero
  * Update the weights based on the loss
  * Update the gradient

### Training result

Loss function = Cross Entropy
Optimizer = Adam
Epochs = 300

```
epoch:   1 loss:    0.78524
epoch:  21 loss:    0.60246
epoch:  41 loss:    0.54884
epoch:  61 loss:    0.50174
epoch:  81 loss:    0.45219
epoch: 101 loss:    0.42238
epoch: 121 loss:    0.39483
epoch: 141 loss:    0.38669
epoch: 161 loss:    0.38241
epoch: 181 loss:    0.38472
epoch: 201 loss:    0.36969
epoch: 221 loss:    0.37332
epoch: 241 loss:    0.36577
epoch: 261 loss:    0.36432
epoch: 281 loss:    0.36342
epoch: 300 loss:    0.35960
```

* After around the 200th epoch, there is a very little decrease in the loss.

## Test the model

### Test result

```
loss: 0.36882
```

* The loss on the test set is slightly more that on the training set
* This implies that our model is slightly overfitting.

## Make predictions

* Predictions are a list
  * Higher value in the first index implies output is 0
  * Higher value in the second index implies output is 1
* Retrieve the index of the larger value in the list to get the required output

## Evaluate the model

Using:

* Confusion matrix
* Accuracy score
* Classification report

The model achieves an accuracy of 84.1%.

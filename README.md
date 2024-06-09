## Train/Test split

You are provided with a full list of real estate properties in three counties (Los Angeles, Orange and Ventura, California) data in 2016.

- **Train Data**:  
  - All transactions before October 15, 2016.
  - Some transactions after October 15, 2016.

- **Test Data (Public Leaderboard)**:  
  - Rest of the transactions between October 15 and December 31, 2016.

You are asked to predict 6 time points for all properties:  
- October 2016 (201610)  
- November 2016 (201611)  
- December 2016 (201612)  

Not all properties are sold in each time period. If a property was not sold in a certain time period, that particular row will be ignored when calculating your score.

If a property is sold multiple times within 31 days, the first reasonable value is taken as the ground truth. By "reasonable," it is meant that if the data seems incorrect, the transaction with a value that makes more sense will be considered.

## File Descriptions

- **properties_2016.csv**:  
  - Contains all the properties with their home features for the year 2016.  

- **train_2016.csv**:  
  - Training set with transactions from 1/1/2016 to 12/31/2016.

- **sample_submission.csv**:  
  - A sample submission file in the correct format.

## Data Fields

Please refer to the "data_dictionary.xlsx" for details on the data fields.

import pandas as pd
import numpy as np
import tensorflow as tf
from network_details import df
from network_details import encoder

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
df['service'] = encoder.fit_transform(df['service'])
df['flag'] = encoder.fit_transform(df['flag'])
df['output'] = encoder.fit_transform(df['output'])

# Create a function to preprocess a single record
def preprocess_record(record):
    # Convert the record to a pandas dataframe
    record_df = pd.DataFrame([record], columns=df.columns[:-1])

    # Encode the categorical variables in the record
    record_df['protocol_type'] = encoder.fit_transform(record_df['protocol_type'])
    record_df['service'] = encoder.fit_transform(record_df['service'])
    record_df['flag'] = encoder.fit_transform(record_df['flag'])
    #tf.convert_to_tensor(record_df)
    records = np.asarray(record_df.values).astype(np.float64)
    # Return the preprocessed record as a numpy array
    return tf.convert_to_tensor(records)


# Take a single record as input
record = np.array([

0,'icmp','eco_i','SF',8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,31,0.00,0.00,0.00,0.00,1.00,0.00,1.00,1,48,1.00,0.00,1.00,0.52,0.00,0.00,0.00,0.00 #anomaly
])

# Preprocess the record
preprocessed_record = preprocess_record(record)

# Make a prediction on the preprocessed record
prediction = model.predict(preprocessed_record)
print(prediction)
if(prediction<=0.5):
    print('anomaly')
else:
    print('normal')


from sklearn import metrics
import pandas as pd

def rec_results(model, X_test, y_test, test_meta, pos_label = 'p', true_col = 'Type', pred_col = 'Pred'):
	# Inputs: model - the model on which to evaluate
	# X_test - test features as a dataframe
    # y_test - the correct labels for each recording
	# test_meta - demographic data for the recordings
    # pos_label - the label used for the posistive class
    # true_col - the name of the column containing the correct label for each recording
    # pred_col - the columns name where the predictions will be stored in the dataframe
	# Outputs: test_meta - a dtaframe containing the demographic data for each recording and the prediction from the model
    # acc - the accuracy of the predictions
    # precision - the precision of the predictions
    # recall - the recall of the predictions
    
    # predict
    y_pred = model.predict(X_test)

    #format results
    results = y_test.to_frame()
    results[pred_col] = y_pred
    test_meta = test_meta.join(results.drop(true_col, axis=1))
    test_meta[pred_col] = test_meta[pred_col].fillna(pos_label)
    
    acc = metrics.accuracy_score(test_meta[true_col], test_meta[pred_col])
    precision = metrics.precision_score(test_meta[true_col], test_meta[pred_col], pos_label=pos_label)
    recall = metrics.recall_score(test_meta[true_col], test_meta[pred_col], pos_label=pos_label)

    return test_meta, acc, precision, recall

def patient_results(model_results, pos_label = 'p', true_col = 'Type', pred_col = 'Pred'):
	# Input: model_results - the dataframe containing the predictions from the model (can be taken straight from the output of rec_results)
    # Outputs: results - a doataframe containing the predictions made for each patient as well as the most common prediction across the three sounds
    # acc - accuracy calculated for the most common prediction for each patient
    # precision - precision calculated for the most common prediction for each patient
    # recall - recall calculated for the most common prediction for each patient
    results = pd.DataFrame(columns = ['RecordingID', 'Type', 'Pred_a', 'Pred_i', 'Pred_u'])
    # get results for each patient
    for idx, row in model_results.iterrows():
        rec_id = row['RecordingID']
        sound = row['Sound']
        pred = row[pred_col]
        if rec_id in list(results['RecordingID']):
            row_index = results.index[results['RecordingID'] == rec_id]
            if sound == 'i':
                    results.loc[row_index,['Pred_i']] = pred  
            elif sound == 'u':
                    results.loc[row_index,['Pred_u']] = pred  
        else:
            results.loc[len(results)] = [rec_id, row[true_col], pred, None, None]

    results['Pred_mode'] = results[['Pred_a', 'Pred_i', 'Pred_u']].mode(axis=1)
    
    acc = metrics.accuracy_score(results[true_col], results['Pred_mode'])
    precision = metrics.precision_score(results[true_col], results['Pred_mode'], pos_label=pos_label)
    recall = metrics.recall_score(results[true_col], results['Pred_mode'], pos_label=pos_label)
    
    return results, acc, precision, recall
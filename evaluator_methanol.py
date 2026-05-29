import json
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.ner_labelling.post_process_entities import PostProcessorEntities
from src.ner_labelling.post_process_relations import PostProcessorRelations
from scipy.optimize import linear_sum_assignment
import os 
import re
import numpy as np

class NERAndRelationEvaluation:
    def __init__(self, input_path, output_path, mode, embedding_model_path):
        self.input_path = input_path
        self.output_path = output_path 
        self.mode = mode
        self.df = pd.read_csv(self.input_path)
        #self.extracted_data = None
        self.entity_processor = PostProcessorEntities(self.df)
        self.relation_processor = PostProcessorRelations(self.df)
        self.embedded_model = SentenceTransformer(embedding_model_path, trust_remote_code=True)
    
    def get_embedding_sentence_transformers(self, embed_model, texts):
        embeddings = embed_model.encode(texts)
        dimensions = 768
        return embeddings[:, :dimensions]

    def fuzzy_match_outbreak(self, prediction, actual, threshold=0.9):
        actual_lower = [item.lower() for item in actual]
        predicted_lower = [item.lower() for item in prediction]
        common_lower = set(actual_lower).intersection(predicted_lower)
        actual_filtered = [item for item in actual if item.lower() not in common_lower]
        predicted_filtered = [item for item in prediction if item.lower() not in common_lower]
        if not predicted_filtered or not actual_filtered:
            return prediction, np.array([[]])

        prediction_embedding = self.get_embedding_sentence_transformers(self.embedded_model, predicted_filtered)
        actual_embedding = self.get_embedding_sentence_transformers(self.embedded_model, actual_filtered)
        cosine_sim_matrix = cosine_similarity(prediction_embedding, actual_embedding)
        cosine_sim_matrix_copy = np.copy(cosine_sim_matrix)

        for row_idx, row in enumerate(cosine_sim_matrix):
            for col_idx, col in enumerate(row):
                if cosine_sim_matrix[row_idx][col_idx] >= threshold:
                    cosine_sim_matrix[row_idx][col_idx] = 1

        row_idx, col_idx = linear_sum_assignment(-cosine_sim_matrix)
        term_arr = []
        for i in range(len(row_idx)):
            if cosine_sim_matrix[row_idx[i], col_idx[i]] == 1:
                term_arr.append(actual_filtered[col_idx[i]])
            else:
                term_arr.append(predicted_filtered[row_idx[i]])

        remaining_idx = set(range(len(predicted_filtered))) - set(row_idx)
        for idx in remaining_idx:
            term_arr.append(predicted_filtered[idx])
        common_original = [item for item in actual if item.lower() in common_lower]
        term_arr.extend(common_original)
        return term_arr, cosine_sim_matrix_copy
    
    def fuzzy_match(self, prediction, actual, threshold=1.0):
        actual_lower = [item.lower() for item in actual]
        predicted_lower = [item.lower() for item in prediction]
        common_lower = set(actual_lower).intersection(predicted_lower)
        actual_filtered = [item for item in actual if item.lower() not in common_lower]
        predicted_filtered = [item for item in prediction if item.lower() not in common_lower]
        if not predicted_filtered or not actual_filtered:
            return prediction

        prediction_embedding = self.get_embedding_sentence_transformers(self.embedded_model, predicted_filtered)
        actual_embedding = self.get_embedding_sentence_transformers(self.embedded_model, actual_filtered)
        cosine_sim_matrix = cosine_similarity(prediction_embedding, actual_embedding)

        for row_idx, row in enumerate(cosine_sim_matrix):
            for col_idx, col in enumerate(row):
                if cosine_sim_matrix[row_idx][col_idx] >= threshold:
                    cosine_sim_matrix[row_idx][col_idx] = 1

        row_idx, col_idx = linear_sum_assignment(-cosine_sim_matrix)
        term_arr = []
        for i in range(len(row_idx)):
            if cosine_sim_matrix[row_idx[i], col_idx[i]] == 1:
                term_arr.append(actual_filtered[col_idx[i]])
            else:
                term_arr.append(predicted_filtered[row_idx[i]])

        remaining_idx = set(range(len(predicted_filtered))) - set(row_idx)
        for idx in remaining_idx:
            term_arr.append(predicted_filtered[idx])
        common_original = [item for item in actual if item.lower() in common_lower]
        term_arr.extend(common_original)
        return term_arr

    def calculate_precision(self, true_positive, false_positive):
        if true_positive + false_positive == 0:
            return 0
        return true_positive / (true_positive + false_positive)

    def calculate_recall(self, true_positive, false_negative):
        if true_positive + false_negative == 0:
            return 0
        return true_positive / (true_positive + false_negative)

    def calculate_f_measure(self, precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_outbreak(self, predicted_outbreak, label_outbreak):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        cosine_mat = np.array([[]])
    
        if not predicted_outbreak and not label_outbreak:
            true_positives += 1
        elif not predicted_outbreak:
            false_negatives += len(label_outbreak)
        elif not label_outbreak:
            false_positives += len(predicted_outbreak)
        else:
            fuzzy_matched_prediction, cosine_mat = self.fuzzy_match_outbreak(predicted_outbreak, label_outbreak)
            predicted_outbreak_set = set(fuzzy_matched_prediction)
            labelled_outbreak_set = set(label_outbreak)
            predicted_outbreak_set = {
                outbreak.lower() for outbreak in predicted_outbreak_set
            }
            labelled_outbreak_set = {
                outbreak.lower() for outbreak in labelled_outbreak_set
            }
            true_positives = len(predicted_outbreak_set & labelled_outbreak_set)
            false_positives = len(predicted_outbreak_set - labelled_outbreak_set)
            false_negatives = len(labelled_outbreak_set - predicted_outbreak_set)

        metrics = {
                "TP": true_positives,
                "FP": false_positives,
                "FN": false_negatives
            }
        return metrics, cosine_mat

    def evaluate_ner(self, predicted_ner, label_ner):
        """
        considered_entities_for_all = [
            "LOCATION",
            "DATE",
            "TIMEFRAME",
            "FATALITY COUNT",
            "CASE NUMBER",
        ]
        """
        considered_entities_for_all = list(set(predicted_ner.keys()) | set(label_ner.keys()))
        metrics = {}

        for entity_type in considered_entities_for_all:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0

            if entity_type not in predicted_ner and entity_type  not in label_ner:
                true_negatives += 1
            elif entity_type not in predicted_ner:
                false_negatives += len(label_ner[entity_type])
            elif entity_type not in label_ner:
                false_positives += len(predicted_ner[entity_type])
            elif any(item.lower() == 'not mentioned' for item in predicted_ner[entity_type]) and any(item.lower() == 'not mentioned' for item in label_ner[entity_type]):
                true_negatives += 1
            elif any(item.lower() == 'not mentioned' for item in predicted_ner[entity_type]):
                true_negatives += len(label_ner[entity_type])
            elif any(item.lower() == 'not mentioned' for item in label_ner[entity_type]):
                false_positives += len(predicted_ner[entity_type])
            else:
                fuzzy_matched_prediction = self.fuzzy_match(predicted_ner[entity_type], label_ner[entity_type])
                predicted_entity_set = set(fuzzy_matched_prediction)
                labelled_entity_set = set(label_ner[entity_type])
                predicted_entity_set = {
                    entity.lower() for entity in predicted_entity_set
                }
                labelled_entity_set = {
                    entity.lower() for entity in labelled_entity_set
                }
                true_positives = len(predicted_entity_set & labelled_entity_set)
                false_positives = len(predicted_entity_set - labelled_entity_set)
                false_negatives = len(labelled_entity_set - predicted_entity_set)

            metrics[entity_type] = {
                "TP": true_positives,
                "FP": false_positives,
                "FN": false_negatives,
                "TN": true_negatives
            }

        return metrics

    def evaluate_relations(self, predicted_re, label_re):
        considered_relations_for_all = [
            "LOCATED AT",
            "FATALITY FROM",
            "CASES OF",
            "FATALITIES IN",
            "CASES IN",
            "OCCURRED ON",
            "OCCURRED WITHIN",
            "SYMPTOMS OF",
        ]
        metrics = {}

        for relation_type in considered_relations_for_all:
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            if relation_type not in predicted_re and relation_type not in label_re:
                true_positives += 1
            elif relation_type not in predicted_re:
                false_negatives += len(label_re[relation_type])
            elif relation_type not in label_re:
                false_positives += len(predicted_re[relation_type])
            else:
                predicted_re_strings = [
                    ", ".join(f"{key}: {value}" for key, value in pr.items())
                    for pr in predicted_re[relation_type]
                ]
                labelled_re_strings = [
                    ", ".join(f"{key}: {value}" for key, value in pr.items())
                    for pr in label_re[relation_type]
                ]

                fuzzy_matched_prediction = self.fuzzy_match(predicted_re_strings, labelled_re_strings)
                predicted_relation_set = set(fuzzy_matched_prediction)
                labelled_relation_set = set(labelled_re_strings)
                predicted_relation_set = {
                    entity.lower() for entity in predicted_relation_set
                }
                labelled_relation_set = {
                    entity.lower() for entity in labelled_relation_set
                }
                true_positives = len(predicted_relation_set & labelled_relation_set)
                false_positives = len(predicted_relation_set - labelled_relation_set)
                false_negatives = len(labelled_relation_set - predicted_relation_set)

            metrics[relation_type] = {
                "TP": true_positives,
                "FP": false_positives,
                "FN": false_negatives
            }
        
        return metrics

    def fit(self):
        full_evaluation = {}
        full_evaluation_ner = {}
        cumulative_evalation = {}

        evaluated_count = 0
        prediction_columns = [
            col for col in self.df.columns 
            if 'feedback' not in col.lower() and 'url' not in col.lower() and '_id' not in col and 'actual' not in col.lower() and 'summary' not in col.lower() and 'entities_dictionary' not in col.lower() and 'relations_dictionary' not in col.lower()
        ]

        for col in prediction_columns:
            full_evaluation[col] = {}
            full_evaluation_ner[col] = {}

        for key, value in self.df.iterrows():
            evaluated_count += 1
            if evaluated_count % 100 == 1:
                print(f"Starting row {evaluated_count}")

            for col in prediction_columns:
                actual = value["feedback"]
                prediction = value[col]
                print(prediction)
                if self.mode == "NER":
                    actual_entities_dictionary = {} if pd.isna(actual) else self.entity_processor.convert_entities_to_dictionary(actual)
                    predicted_entities_dictionary = {} if pd.isna(prediction) else self.entity_processor.convert_entities_to_dictionary(prediction)
                    print(predicted_entities_dictionary)
                    evaluation = self.evaluate_ner(predicted_entities_dictionary, actual_entities_dictionary)
                else:
                    actual_relations_dictionary = {} if pd.isna(actual) else self.relation_processor.convert_relations_to_dictionary(actual)[0]
                    predicted_relations_dictionary = {} if pd.isna(prediction) else self.relation_processor.convert_relations_to_dictionary(prediction)[0]
                    evaluation = self.evaluate_relations(predicted_relations_dictionary, actual_relations_dictionary)

                for eval_type in evaluation:
                    if eval_type not in full_evaluation[col]:
                        full_evaluation[col][eval_type] = evaluation[eval_type]
                        full_evaluation_ner[col][eval_type] = {}
                    else:
                        full_evaluation[col][eval_type]['TP'] += evaluation[eval_type]['TP']
                        full_evaluation[col][eval_type]['FP'] += evaluation[eval_type]['FP']
                        full_evaluation[col][eval_type]['FN'] += evaluation[eval_type]['FN']
                        full_evaluation[col][eval_type]['TN'] += evaluation[eval_type]['TN']

                    precision = self.calculate_precision(full_evaluation[col][eval_type]['TP'], full_evaluation[col][eval_type]['FP'])
                    recall = self.calculate_recall(full_evaluation[col][eval_type]['TP'], full_evaluation[col][eval_type]['FN'])
                    f_measure = self.calculate_f_measure(precision, recall)
                    if evaluated_count not in cumulative_evalation:
                        cumulative_evalation[evaluated_count] = {}
                    if col not in cumulative_evalation[evaluated_count]:
                        cumulative_evalation[evaluated_count][col] = {}
                    cumulative_evalation[evaluated_count][col][eval_type] = f_measure

        rows = []
        for row, methods in cumulative_evalation.items():
            for method, entities in methods.items():
                row_data = {'row': row, 'method': method}
                row_data.update(entities)
                rows.append(row_data)
        cumulative_df = pd.DataFrame(rows)

        for col in prediction_columns:
            for eval_type in full_evaluation[col]:
                precision = self.calculate_precision(full_evaluation[col][eval_type]['TP'], full_evaluation[col][eval_type]['FP'])
                recall = self.calculate_recall(full_evaluation[col][eval_type]['TP'], full_evaluation[col][eval_type]['FN'])
                full_evaluation_ner[col][eval_type]['TP'] = full_evaluation[col][eval_type]['TP']
                full_evaluation_ner[col][eval_type]['FP'] = full_evaluation[col][eval_type]['FP']
                full_evaluation_ner[col][eval_type]['FN'] = full_evaluation[col][eval_type]['FN']
                full_evaluation_ner[col][eval_type]['TN'] = full_evaluation[col][eval_type]['TN']
                full_evaluation_ner[col][eval_type]['precision'] = precision
                full_evaluation_ner[col][eval_type]['recall'] = recall
                full_evaluation_ner[col][eval_type]['fmeasure'] = self.calculate_f_measure(precision, recall)

        df_rows = []
        for method, entities in full_evaluation_ner.items():
            method_name = method
            for entity, metrics in entities.items():
                row = {
                    'ENTITY_TYPE': entity,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'fmeasure': metrics['fmeasure'],
                    'method': method_name,
                    'TP': metrics['TP'],
                    'FP': metrics['FP'],
                    'FN': metrics['FN'],
                    'TN': metrics['TN']
                }
                df_rows.append(row)

        df = pd.DataFrame(df_rows)
        return df, cumulative_df

if __name__ == "__main__":
    input_path = ""
    output_path = ""
    

    NER_RE_evaluation = NERAndRelationEvaluation(input_path, output_path, "NER", "")
    out, cumulative_out = NER_RE_evaluation.fit()
    out.to_csv(output_path, index=False)
cumulative_long = (
    cumulative_out
      .melt(id_vars=["row","method"], var_name="ENTITY_TYPE", value_name="f1")
      .dropna(subset=["f1"])
)

cumulative_out_path = output_path.replace(".csv", "_cumulative.csv")
cumulative_long.to_csv(cumulative_out_path, index=False)
print("Saved:", cumulative_out_path)

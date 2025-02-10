import json
import re
import math
import os  # For file existence checking.
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK stopwords are downloaded.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Initialize the stemmer.
ps = PorterStemmer()

# --- Global Variable for the Generated Symptoms JSON File ---
GENERATED_SYMPTOMS_FILE = "generated_symptoms.json"

# --- New Functions for Saving/Loading Generated Symptom Data ---

def save_generated_symptom_data(illness, symptoms):
    """
    Save (or update) a record in GENERATED_SYMPTOMS_FILE.

    If a record with the same 'Illness' already exists,
    overwrite it with the new list of symptoms. Otherwise, add a new record.
    """
    record = {"Illness": illness, "Symptoms": symptoms}
    data = []
    if os.path.exists(GENERATED_SYMPTOMS_FILE):
        try:
            with open(GENERATED_SYMPTOMS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    # Check if a record for the same illness exists; if so, update it.
    replaced = False
    for i, rec in enumerate(data):
        if rec.get("Illness") == illness:
            data[i] = record
            replaced = True
            break
    if not replaced:
        data.append(record)
    with open(GENERATED_SYMPTOMS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_generated_symptoms():
    """
    Loads any previously saved generated symptom data from the JSON file
    and adds them to the training data.
    """
    if os.path.exists(GENERATED_SYMPTOMS_FILE):
        try:
            with open(GENERATED_SYMPTOMS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
        for record in data:
            symptoms = record.get("Symptoms", [])
            illness = record.get("Illness", "")
            if symptoms and illness:
                query_text = " ".join(symptoms).lower()
                training_queries.append(query_text)
                training_labels.append(illness)

def update_common_symptoms():
    """
    Update the global common_symptoms list by merging symptoms extracted from
    the medication data with those stored in the generated_symptoms.json file.
    This updated list is then set as the values for the symptom combobox.
    """
    global common_symptoms
    # Start with symptoms from the medication data.
    base_symptoms = set(extract_symptoms(medication_data))
    # Add any generated symptoms from the JSON file.
    if os.path.exists(GENERATED_SYMPTOMS_FILE):
        try:
            with open(GENERATED_SYMPTOMS_FILE, "r", encoding="utf-8") as f:
                gen_data = json.load(f)
        except json.JSONDecodeError:
            gen_data = []
        for record in gen_data:
            for symptom in record.get("Symptoms", []):
                s = symptom.strip()
                if s:
                    base_symptoms.add(s)
    common_symptoms = sorted(base_symptoms)
    symptom_dropdown_new["values"] = common_symptoms

# --- Data Loading and Utility Functions ---

def load_medication_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "medications" in data:
            return data["medications"]
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data from {filename}:\n{e}")
        return {}

def extract_symptoms(med_data):
    """
    Extracts and returns a sorted list of unique symptoms from the medication data.
    """
    symptoms_set = set()
    for category, details in med_data.items():
        if "Symptoms" in details:
            symptom_list = details["Symptoms"]
            if isinstance(symptom_list, list):
                for symptom in symptom_list:
                    symptoms_set.add(symptom.strip())
            elif isinstance(symptom_list, str):
                symptoms_set.update(s.strip() for s in symptom_list.split(","))
    return sorted(symptoms_set)

# --- TF–IDF & BM25 Functions with Stemming & Stopword Removal ---

def compute_tf(doc, stemmer):
    """Compute normalized term frequency for a document using stemmed tokens and stopword removal."""
    tokens = re.findall(r'\b\w+\b', doc.lower())
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    total = len(tokens)
    if total > 0:
        for token in tf:
            tf[token] /= total
    return tf

def compute_df(doc_tf):
    """Compute document frequency from a dictionary of TF dictionaries."""
    df = {}
    for tf_dict in doc_tf.values():
        for term in tf_dict:
            df[term] = df.get(term, 0) + 1
    return df

def compute_idf(df, N):
    """Compute inverse document frequency."""
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N + 1) / (freq + 1)) + 1
    return idf

def compute_tfidf(tf, idf):
    """Compute the TF–IDF vector from TF and IDF."""
    tfidf = {}
    for token, freq in tf.items():
        tfidf[token] = freq * idf.get(token, 0)
    return tfidf

def cosine_similarity_tfidf(vec1, vec2):
    """Compute cosine similarity between two TF–IDF vectors."""
    dot = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)
    norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

def compute_query_tfidf(query_list, idf, stemmer):
    """Compute the TF–IDF vector for the query (a list of selected symptoms) with stopword removal."""
    query_text = " ".join(query_list).lower()
    tokens = re.findall(r'\b\w+\b', query_text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    total = len(tokens)
    if total > 0:
        for token in tf:
            tf[token] /= total
    tfidf = {}
    for token, freq in tf.items():
        tfidf[token] = freq * idf.get(token, 0)
    return tfidf

def get_stemmed_token_set(text, stemmer):
    """Return a set of stemmed tokens from text."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(stemmer.stem(token) for token in tokens if token not in stop_words)

# Global variables for TF–IDF and BM25
illness_documents = {}  # illness -> document text
doc_tf = {}  # illness -> normalized TF dictionary (stemmed)
df = {}  # document frequency for terms
idf = {}  # inverse document frequency for terms
illness_tfidf = {}  # illness -> TF–IDF vector
doc_tf_raw = {}  # illness -> raw term frequency dictionary (stemmed)
avg_doc_len = 0  # average document length (raw token count)

def build_tfidf_vectors():
    global illness_documents, doc_tf, df, idf, illness_tfidf, doc_tf_raw, avg_doc_len
    illness_documents = {}
    doc_tf = {}
    doc_tf_raw = {}
    total_doc_len = 0
    for illness, details in medication_data.items():
        text = ""
        if "CategoryDescription" in details:
            text += details["CategoryDescription"] + " "
        if "Symptoms" in details:
            if isinstance(details["Symptoms"], list):
                text += " " + " ".join(details["Symptoms"]) + " "
            else:
                text += " " + details["Symptoms"] + " "
        for key, val in details.items():
            if isinstance(val, dict):
                for med, med_details in val.items():
                    if "Description" in med_details:
                        text += med_details["Description"] + " "
        text = text.lower()
        illness_documents[illness] = text
        # Normalized TF for TF–IDF
        tf_norm = compute_tf(text, ps)
        doc_tf[illness] = tf_norm
        # Raw TF for BM25
        tokens = re.findall(r'\b\w+\b', text)
        tokens = [ps.stem(token) for token in tokens if token not in stop_words]
        raw_tf = {}
        for token in tokens:
            raw_tf[token] = raw_tf.get(token, 0) + 1
        doc_tf_raw[illness] = raw_tf
        doc_len = sum(raw_tf.values())
        total_doc_len += doc_len
    # Compute document frequency and IDF.
    df = compute_df(doc_tf)
    N = len(doc_tf)
    idf.clear()
    idf.update(compute_idf(df, N))
    # Build TF–IDF vectors.
    illness_tfidf.clear()
    for illness in doc_tf:
        illness_tfidf[illness] = compute_tfidf(doc_tf[illness], idf)
    avg_doc_len = total_doc_len / N if N > 0 else 0

def compute_bm25_score(illness, query_tokens, k1=1.5, b=0.75):
    """Compute the BM25 score for an illness document given a set of query tokens."""
    raw_tf = doc_tf_raw.get(illness, {})
    doc_len = sum(raw_tf.values())
    score = 0
    for term in query_tokens:
        if term in raw_tf:
            tf_term = raw_tf[term]
            numerator = tf_term * (k1 + 1)
            denominator = tf_term + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf.get(term, 0) * (numerator / denominator)
    return score

# --- Supervised Learning: Training Data & Classifier ---
# Global variables for training data and the classifier.
training_queries = []  # List of query strings (concatenated symptoms)
training_labels = []  # Corresponding chosen illness (string)
classifier_model = None  # The trained classifier (LogisticRegression)
vectorizer_model = None  # The TF–IDF vectorizer for queries

def train_classifier_model():
    """Train a Logistic Regression classifier on the training data (if sufficient data exists)."""
    global classifier_model, vectorizer_model
    if len(training_queries) < 5:
        # Not enough data to train
        return
    vectorizer_model = TfidfVectorizer()
    X_train = vectorizer_model.fit_transform(training_queries)
    classifier_model = LogisticRegression(max_iter=1000)
    classifier_model.fit(X_train, training_labels)

# --- Callback Functions for the Main Treeview ---

def get_subtree_text(item_id, indent=0):
    result = " " * indent + tree.item(item_id, "text")
    for child in tree.get_children(item_id):
        result += "\n" + get_subtree_text(child, indent + 2)
    return result

def copy_entire_tree():
    lines = []
    for root_item in tree.get_children(""):
        lines.append(get_subtree_text(root_item))
    full_text = "\n".join(lines)
    root.clipboard_clear()
    root.clipboard_append(full_text)
    messagebox.showinfo("Copied", "The entire tree has been copied to the clipboard.")

def on_tree_select(event):
    selected_item = tree.focus()
    if not selected_item:
        return
    text_output.config(state=tk.NORMAL)
    text_output.delete("1.0", tk.END)
    if selected_item in medication_details:
        details = medication_details[selected_item]
        details_str = ""
        if "Illness" in details:
            details_str += f"Category: {details['Illness']}\n\n"
        order = ["Description", "Medicine", "Form", "Dosage", "AgeGroup", "Sex",
                 "Availability", "Preparation", "Frequency", "Notes", "Alternatives", "Maximum"]
        for key in order:
            if key in details:
                value = details[key] if details[key] is not None else "N/A"
                if isinstance(value, str):
                    if key == "Alternatives":
                        alternatives = [f"- {alt.strip()}" for alt in value.split(',') if alt.strip()]
                        value_formatted = "\n".join(alternatives)
                    else:
                        sentences = [s.strip() for s in value.split('.') if s.strip()]
                        value_formatted = "\n".join(sentence + "." for sentence in sentences)
                else:
                    value_formatted = str(value)
                details_str += f"{key}: {value_formatted}\n\n"
        text_output.insert(tk.END, details_str.strip(), "center")
    else:
        node_text = tree.item(selected_item, "text")
        if node_text in medication_data and "CategoryDescription" in medication_data[node_text]:
            cat_desc = medication_data[node_text]["CategoryDescription"].strip()
            cat_desc_lines = [line.strip() for line in cat_desc.split('.') if line.strip()]
            cat_desc_output = "\n\n".join(line + "." for line in cat_desc_lines)
            output_text = cat_desc_output
            if "Symptoms" in medication_data[node_text]:
                symptoms = medication_data[node_text]["Symptoms"]
                if isinstance(symptoms, list):
                    symptoms_items = [item.strip() for item in symptoms if item.strip()]
                else:
                    symptoms_items = [s.strip() for s in symptoms.split(',') if s.strip()]
                symptoms_output = "\n\n".join(symptoms_items)
                output_text += "\n\n" + symptoms_output
            text_output.insert(tk.END, output_text, "center")
        else:
            text_output.insert(tk.END, "Please select a specific medication or remedy to view its details.", "center")
    text_output.config(state=tk.DISABLED)

# --- Auto-suggestion Functions for the Main Treeview ---

def populate_all_nodes(parent=""):
    nodes = []
    for item in tree.get_children(parent):
        nodes.append((item, tree.item(item, "text")))
        nodes.extend(populate_all_nodes(item))
    return nodes

def update_suggestions(*args):
    query = search_var.get().strip().lower()
    suggestion_listbox.delete(0, tk.END)
    global suggestion_mapping
    suggestion_mapping = {}
    if query == "":
        return
    for item_id, text in all_nodes:
        if text.lower().startswith(query):
            suggestion_listbox.insert(tk.END, text)
            suggestion_mapping[text] = item_id

def on_suggestion_select(event):
    selection = suggestion_listbox.curselection()
    if selection:
        selected_text = suggestion_listbox.get(selection[0])
        if selected_text in suggestion_mapping:
            item_id = suggestion_mapping[selected_text]
            tree.selection_set(item_id)
            tree.focus(item_id)
            tree.see(item_id)
            search_var.set(selected_text)
            suggestion_listbox.delete(0, tk.END)

# --- New Functions for Symptom-Based Illness Prediction & Training ---

selected_symptoms_global = []
illness_feedback = {}  # Self-learning feedback counts

def add_symptom_new():
    sym = symptom_var_new.get().strip()
    if sym.lower() == "enter symptom..." or sym == "":
        return
    if sym and sym not in selected_symptoms_global:
        selected_symptoms_global.append(sym)
        symptoms_listbox_new.insert(tk.END, sym)
    symptom_var_new.set("")
    update_symptom_autocomplete()
    predict_illness_new()

def remove_symptom_new():
    selected_indices = symptoms_listbox_new.curselection()
    if not selected_indices:
        return
    for index in reversed(selected_indices):
        symptom = symptoms_listbox_new.get(index)
        if symptom in selected_symptoms_global:
            selected_symptoms_global.remove(symptom)
        symptoms_listbox_new.delete(index)
    predict_illness_new()

def predict_illness_new():
    """
    Enhanced prediction algorithm that:
      - Computes a hybrid unsupervised score combining cosine similarity, match ratio, and BM25.
      - Normalizes BM25 scores across all illnesses.
      - Applies a feedback boost.
      - Incorporates supervised classifier probability (if available).
      - Uses softmax with a temperature parameter to produce a balanced probability distribution.
    """
    # Tunable weight parameters.
    alpha = 0.8  # Weight for cosine similarity.
    beta = 0.2  # Weight for match ratio.
    gamma = 0.3  # Weight for BM25 (after normalization) relative to the hybrid score.
    lambda_model = 0.3  # Weight for the classifier's probability.
    score_scale = 100  # Overall scale for raw scores.
    temperature = 10.0  # Temperature for softmax (increase to flatten the distribution).

    # Build query representation.
    query_text = " ".join(selected_symptoms_global).lower()
    query_vector = compute_query_tfidf(selected_symptoms_global, idf, ps)
    query_tokens = set(
        ps.stem(token) for token in re.findall(r'\b\w+\b', query_text)
        if token not in stop_words
    )

    raw_scores = {}
    bm25_scores = {}
    hybrid_scores = {}

    # First, compute BM25 and hybrid scores for each illness.
    for illness in medication_data:
        # Cosine similarity and match ratio.
        cosine_sim = cosine_similarity_tfidf(query_vector, illness_tfidf.get(illness, {}))
        doc_token_set = set(doc_tf[illness].keys())
        match_ratio = (len(query_tokens.intersection(doc_token_set)) / len(query_tokens)) if query_tokens else 0
        hybrid_score = alpha * cosine_sim + beta * match_ratio
        hybrid_scores[illness] = hybrid_score

        # BM25 score (before normalization).
        bm25_scores[illness] = compute_bm25_score(illness, query_tokens)

    # Normalize BM25 scores using min–max normalization.
    if bm25_scores:
        min_bm25 = min(bm25_scores.values())
        max_bm25 = max(bm25_scores.values())
    else:
        min_bm25, max_bm25 = 0, 0

    for illness in medication_data:
        if max_bm25 > min_bm25:
            normalized_bm25 = (bm25_scores[illness] - min_bm25) / (max_bm25 - min_bm25)
        else:
            normalized_bm25 = 0
        # Combine the normalized BM25 with the hybrid score.
        unsupervised_score = gamma * normalized_bm25 + (1 - gamma) * hybrid_scores[illness]
        # Apply a feedback boost.
        feedback_factor = 1 + math.log(illness_feedback.get(illness, 0) + 1)
        final_score = unsupervised_score * feedback_factor * score_scale
        raw_scores[illness] = final_score

    # Incorporate the supervised classifier's probability (if available).
    if classifier_model is not None and vectorizer_model is not None:
        X_query = vectorizer_model.transform([query_text])
        probas = classifier_model.predict_proba(X_query)[0]  # Vector of probabilities.
        classes = classifier_model.classes_
        for illness in raw_scores:
            if illness in classes:
                idx = list(classes).index(illness)
                model_prob = probas[idx]
            else:
                model_prob = 0
            raw_scores[illness] += lambda_model * model_prob * score_scale

    # Convert raw scores into a probability distribution using softmax with temperature.
    illnesses = list(raw_scores.keys())
    scores_array = np.array([raw_scores[ill] for ill in illnesses])
    # Subtract max for numerical stability.
    exp_scores = np.exp((scores_array - np.max(scores_array)) / temperature)
    softmax_scores = exp_scores / np.sum(exp_scores)
    normalized_scores = {illness: softmax_scores[i] * 100 for i, illness in enumerate(illnesses)}

    # Sort and display the top 10 illnesses.
    sorted_illnesses = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    for item in predicted_tree.get_children():
        predicted_tree.delete(item)
    for illness, score in sorted_illnesses[:10]:
        predicted_tree.insert("", tk.END, values=(illness, f"{score:.2f}%"))

def on_predicted_tree_select(event):
    selection = predicted_tree.selection()
    if selection:
        item = predicted_tree.selection()[0]
        predicted_values = predicted_tree.item(item, "values")
        if predicted_values:
            illness_name = predicted_values[0]
            # Update self-learning feedback.
            if illness_name in illness_feedback:
                illness_feedback[illness_name] += 1
            else:
                illness_feedback[illness_name] = 1
            # Add this query and label to the training data.
            query_text = " ".join(selected_symptoms_global).lower()
            if query_text and illness_name:
                training_queries.append(query_text)
                training_labels.append(illness_name)
                train_classifier_model()  # Retrain classifier with updated data

                # Save or update the generated symptom data (overwrites if the same illness exists)
                save_generated_symptom_data(illness_name, selected_symptoms_global)
                # Also update the combobox list with new symptoms.
                update_common_symptoms()

            # Expand the tree to show the selected illness.
            for node_id, text in all_nodes:
                if text.lower() == illness_name.lower():
                    parent_id = tree.parent(node_id)
                    while parent_id:
                        tree.item(parent_id, open=True)
                        parent_id = tree.parent(parent_id)
                    tree.selection_set(node_id)
                    tree.focus(node_id)
                    tree.see(node_id)
                    on_tree_select(None)
                    break

# --- Updated Auto-complete for the Symptom Combobox ---

def update_symptom_autocomplete(event=None):
    current_text = symptom_var_new.get().strip().lower()
    if current_text == "" or current_text == "enter symptom...":
        filtered = common_symptoms
    else:
        filtered = [sym for sym in common_symptoms
                    if any(word.startswith(current_text) for word in re.split(r'\W+', sym.lower()) if word)]
        if not filtered:
            filtered = [sym for sym in common_symptoms if current_text in sym.lower()]
    symptom_dropdown_new['values'] = filtered

def on_symptom_double_click(event):
    selection = symptoms_listbox_new.curselection()
    if selection:
        index = selection[0]
        symptom = symptoms_listbox_new.get(index)
        if symptom in selected_symptoms_global:
            selected_symptoms_global.remove(symptom)
        symptoms_listbox_new.delete(index)
    predict_illness_new()

# --- GUI Setup ---

root = tk.Tk()
root.title("Medication Regimen Viewer & Symptom Checker")
root.geometry("1700x900")

paned_window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Left frame: Treeview and search.
tree_frame = ttk.Frame(paned_window, width=500, relief=tk.SUNKEN)
paned_window.add(tree_frame, weight=1)

# Middle frame: Text output.
text_frame = ttk.Frame(paned_window, width=200, relief=tk.SUNKEN)
paned_window.add(text_frame, weight=1)

# Right frame: Symptom input and prediction.
predicted_frame = ttk.Frame(paned_window, width=600, relief=tk.SUNKEN)
paned_window.add(predicted_frame, weight=3)

# --- Left Pane Setup (Treeview & Search) ---
search_frame = ttk.Frame(tree_frame)
search_frame.pack(fill=tk.X, padx=5, pady=5)

search_label = ttk.Label(search_frame, text="Search:")
search_label.pack(side=tk.LEFT, padx=(0, 5))

search_var = tk.StringVar()
search_entry = ttk.Entry(search_frame, textvariable=search_var, width=60)
search_entry.pack(side=tk.LEFT, padx=(0, 5))

suggestion_listbox = tk.Listbox(search_frame, height=5)
suggestion_listbox.pack(fill=tk.X, padx=5, pady=(2, 5))
suggestion_listbox.bind("<<ListboxSelect>>", on_suggestion_select)
search_var.trace("w", update_suggestions)

tree = ttk.Treeview(tree_frame)
tree.heading("#0", text="Category / Treatment / Medication")
tree.column("#0", width=400)
tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
tree.bind("<<TreeviewSelect>>", on_tree_select)

# --- Middle Pane Setup (Text Output) ---
text_output = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED, width=80)
text_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
text_output.tag_configure("center", justify="center")

# --- Right Pane Setup (Symptom Input & Predicted Illness) ---
symptom_input_frame = ttk.Frame(predicted_frame)
symptom_input_frame.pack(fill=tk.X, padx=5, pady=5)
symptom_input_frame.grid_columnconfigure(1, weight=1)

symptom_label_new = ttk.Label(symptom_input_frame, text="Enter Symptom:")
symptom_label_new.grid(row=0, column=0, padx=5, pady=5, sticky="w")

symptom_var_new = tk.StringVar()
symptom_dropdown_new = ttk.Combobox(
    symptom_input_frame,
    textvariable=symptom_var_new,
    values=[],  # To be filled by update_common_symptoms()
    state="normal",
    width=50
)
symptom_dropdown_new.grid(row=0, column=1, padx=5, pady=5, sticky="w")
symptom_dropdown_new.set("")
symptom_dropdown_new.bind("<KeyRelease>", update_symptom_autocomplete)
symptom_dropdown_new.bind("<<ComboboxSelected>>", lambda event: add_symptom_new())

symptoms_listbox_new = tk.Listbox(predicted_frame, height=10, selectmode=tk.EXTENDED)
symptoms_listbox_new.pack(fill=tk.X, padx=5, pady=5)
symptoms_listbox_new.bind("<Double-Button-1>", on_symptom_double_click)

predicted_tree = ttk.Treeview(predicted_frame, columns=("Illness", "Score"), show="headings", height=10)
predicted_tree.heading("Illness", text="Illness")
predicted_tree.heading("Score", text="Score")
predicted_tree.column("Illness", width=200)
predicted_tree.column("Score", width=50)
predicted_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
predicted_tree.bind("<<TreeviewSelect>>", on_predicted_tree_select)

# --- Footer Setup ---
footer_frame = ttk.Frame(root)
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

copy_all_btn = ttk.Button(footer_frame, text="Copy Entire Tree", command=copy_entire_tree)
copy_all_btn.grid(row=0, column=0, padx=5, pady=5)

add_symptom_btn = ttk.Button(footer_frame, text="Add", command=add_symptom_new)
add_symptom_btn.grid(row=0, column=1, padx=5, pady=5)

remove_symptom_btn = ttk.Button(footer_frame, text="Remove", command=remove_symptom_new)
remove_symptom_btn.grid(row=0, column=2, padx=5, pady=5)

predict_btn = ttk.Button(footer_frame, text="Predict Illness", command=predict_illness_new)
predict_btn.grid(row=0, column=3, padx=5, pady=5)

# --- Load JSON Data and Build the Main Tree ---
medication_data = load_medication_data('medication.json')
common_symptoms = extract_symptoms(medication_data)
# Initially set the combobox values from medication data.
symptom_dropdown_new["values"] = common_symptoms
print("Extracted Common Symptoms:", common_symptoms)

medication_details = {}
for category in sorted(medication_data.keys()):
    cat_data = medication_data[category]
    category_id = tree.insert("", tk.END, text=category, open=False)
    for treatment_type, meds in cat_data.items():
        if treatment_type in ["CategoryDescription", "Symptoms"]:
            continue
        treatment_id = tree.insert(category_id, tk.END, text=treatment_type, open=False)
        if isinstance(meds, dict):
            for med_name, details in meds.items():
                med_id = tree.insert(treatment_id, tk.END, text=med_name)
                details_copy = details.copy()
                details_copy["Illness"] = category
                medication_details[med_id] = details_copy

aggregated_medicines_node = tree.insert("", tk.END, text="Medicines", open=False)
for category, treatment_types in medication_data.items():
    if isinstance(treatment_types, dict) and "Conventional" in treatment_types:
        meds = treatment_types["Conventional"]
        if isinstance(meds, dict):
            for med_name, details in meds.items():
                details_copy = details.copy()
                details_copy["Illness"] = category
                med_node = tree.insert(aggregated_medicines_node, tk.END, text=med_name)
                medication_details[med_node] = details_copy

all_nodes = []
def gather_nodes(parent=""):
    for item in tree.get_children(parent):
        all_nodes.append((item, tree.item(item, "text")))
        gather_nodes(item)
gather_nodes()
suggestion_mapping = {}

# --- Build TF-IDF and BM25 Vectors for Prediction ---
build_tfidf_vectors()

# --- Load Previously Generated Symptom Data into the Training Set ---
load_generated_symptoms()
# Update the combobox list with any generated symptoms.
update_common_symptoms()

root.mainloop()

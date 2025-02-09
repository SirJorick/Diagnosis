import json
import re
import math
import tkinter as tk
from tkinter import ttk, messagebox


# --- Data Loading and Utility Functions ---

def load_medication_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # If your JSON has a top-level "medications" key, return its value.
        if "medications" in data:
            return data["medications"]
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data from {filename}:\n{e}")
        return {}


def extract_symptoms(med_data):
    """
    Extracts and returns a sorted list of unique symptoms from the medication data.
    It looks for a "Symptoms" key in each category.
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


# --- Callback Functions for the Main Treeview (Medication Viewer) ---

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

    # If the selected node contains detailed medication info.
    if selected_item in medication_details:
        details = medication_details[selected_item]
        details_str = ""
        if "Illness" in details:
            details_str += f"Category: {details['Illness']}  "
        order = ["Description", "Medicine", "Form", "Dosage", "AgeGroup", "Sex",
                 "Availability", "Preparation", "Frequency", "Notes", "Alternatives", "Maximum"]
        for key in order:
            if key in details:
                value = details[key] if details[key] is not None else "N/A"
                if key == "Alternatives":
                    details_str += f"{key}: "
                    alternatives = [alt.strip() for alt in str(value).split(',') if alt.strip()]
                    details_str += " ".join(f"- {alt}" for alt in alternatives) + "  "
                else:
                    details_str += f"{key}: {value}  "
        text_output.insert(tk.END, details_str.strip())
    else:
        # For category nodes, display the CategoryDescription and Symptoms,
        # separating each sentence and each comma-separated item by double newlines (\n\n).
        node_text = tree.item(selected_item, "text")
        if node_text in medication_data and "CategoryDescription" in medication_data[node_text]:
            cat_desc = medication_data[node_text]["CategoryDescription"].strip()
            # Split the description into sentences based on periods and remove extra whitespace.
            cat_desc_lines = [line.strip() for line in cat_desc.split('.') if line.strip()]
            cat_desc_output = "\n\n".join(line + "." for line in cat_desc_lines)
            output_text = cat_desc_output
            # Process Symptoms if available.
            if "Symptoms" in medication_data[node_text]:
                symptoms = medication_data[node_text]["Symptoms"]
                if isinstance(symptoms, list):
                    symptoms_items = [item.strip() for item in symptoms if item.strip()]
                else:
                    symptoms_items = [s.strip() for s in symptoms.split(',') if s.strip()]
                symptoms_output = "\n\n".join(symptoms_items)
                output_text += "\n\n" + symptoms_output
            text_output.insert(tk.END, output_text)
        else:
            text_output.insert(tk.END, "Please select a specific medication or remedy to view its details.")
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


# --- New Functions for Symptom-Based Illness Prediction ---

selected_symptoms_global = []


def add_symptom_new():
    sym = symptom_var_new.get().strip()
    if sym and sym not in selected_symptoms_global:
        selected_symptoms_global.append(sym)
        symptoms_listbox_new.insert(tk.END, sym)


def remove_symptom_new():
    selected_indices = symptoms_listbox_new.curselection()
    if not selected_indices:
        return
    for index in reversed(selected_indices):
        symptom = symptoms_listbox_new.get(index)
        if symptom in selected_symptoms_global:
            selected_symptoms_global.remove(symptom)
        symptoms_listbox_new.delete(index)


def predict_illness_new():
    """
    Improved prediction algorithm:
      - Concatenate the CategoryDescription, all treatment Description fields,
        and if available, the Symptoms field.
      - Convert text to lowercase and compute total word count.
      - For each selected symptom, count whole-word matches using regex.
      - Apply logarithmic scaling with math.log(count + 1).
      - Compute match ratio = (# of selected symptoms that appear at least once) / (total selected symptoms).
      - Final score = (sum(log(count+1)) / total_words) * match_ratio * 100, expressed as a percentage.
    """
    scores = {}
    for illness, details in medication_data.items():
        text = ""
        if "CategoryDescription" in details:
            text += details["CategoryDescription"] + " "
        # Include the Symptoms field if available.
        if "Symptoms" in details and isinstance(details["Symptoms"], list):
            text += " " + " ".join(details["Symptoms"]) + " "
        for key, val in details.items():
            if isinstance(val, dict):
                for med, med_details in val.items():
                    if "Description" in med_details:
                        text += med_details["Description"] + " "
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words) if words else 1
        weighted_score = 0.0
        match_count = 0
        for sym in selected_symptoms_global:
            count = len(re.findall(r'\b' + re.escape(sym.lower()) + r'\b', text))
            if count > 0:
                match_count += 1
            weighted_score += math.log(count + 1)
        match_ratio = match_count / len(selected_symptoms_global) if selected_symptoms_global else 0
        final_score = (weighted_score / total_words) * match_ratio * 100
        scores[illness] = final_score
    sorted_illnesses = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for item in predicted_tree.get_children():
        predicted_tree.delete(item)
    for illness, score in sorted_illnesses[:3]:
        predicted_tree.insert("", tk.END, values=(illness, f"{score:.2f}%"))


def on_predicted_tree_select(event):
    selection = predicted_tree.selection()
    if selection:
        item = selection[0]
        predicted_values = predicted_tree.item(item, "values")
        if predicted_values:
            illness_name = predicted_values[0]
            # Expand the corresponding node in the left treeview.
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


# --- GUI Setup ---

root = tk.Tk()
root.title("Medication Regimen Viewer & Symptom Checker")
root.geometry("1500x900")

paned_window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Left frame: search, suggestions, treeview, and copy button.
tree_frame = ttk.Frame(paned_window, width=700, relief=tk.SUNKEN)
paned_window.add(tree_frame, weight=1)

# Middle frame: text output for medication details.
text_frame = ttk.Frame(paned_window, width=400, relief=tk.SUNKEN)
paned_window.add(text_frame, weight=3)

# Right frame: for symptom input and predicted illnesses.
predicted_frame = ttk.Frame(paned_window, width=400, relief=tk.SUNKEN)
paned_window.add(predicted_frame, weight=1)

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
tree.column("#0", width=600)
tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

copy_all_btn = ttk.Button(tree_frame, text="Copy Entire Tree", command=copy_entire_tree)
copy_all_btn.pack(padx=5, pady=5)

# --- Middle Pane Setup (Text Output) ---
text_output = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED)
text_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
tree.bind("<<TreeviewSelect>>", on_tree_select)

# --- Right Pane Setup (Symptom Input & Predicted Illness) ---
symptom_input_frame = ttk.Frame(predicted_frame)
symptom_input_frame.pack(fill=tk.X, padx=5, pady=5)

symptom_label_new = ttk.Label(symptom_input_frame, text="Enter Symptom:")
symptom_label_new.pack(side=tk.LEFT, padx=(0, 5))

symptom_var_new = tk.StringVar()
symptom_dropdown_new = ttk.Combobox(symptom_input_frame, textvariable=symptom_var_new, values=[], state="normal")
symptom_dropdown_new.pack(side=tk.LEFT, padx=(0, 5))
symptom_dropdown_new.set("Enter symptom...")

add_symptom_btn = ttk.Button(symptom_input_frame, text="Add", command=add_symptom_new)
add_symptom_btn.pack(side=tk.LEFT, padx=5)

remove_symptom_btn = ttk.Button(symptom_input_frame, text="Remove", command=remove_symptom_new)
remove_symptom_btn.pack(side=tk.LEFT, padx=5)

symptoms_listbox_new = tk.Listbox(predicted_frame, height=10, selectmode=tk.EXTENDED)
symptoms_listbox_new.pack(fill=tk.X, padx=5, pady=5)

predict_btn = ttk.Button(predicted_frame, text="Predict Illness", command=predict_illness_new)
predict_btn.pack(padx=5, pady=5)

predicted_tree = ttk.Treeview(predicted_frame, columns=("Illness", "Score"), show="headings", height=10)
predicted_tree.heading("Illness", text="Illness")
predicted_tree.heading("Score", text="Score")
predicted_tree.column("Illness", width=200)
predicted_tree.column("Score", width=50)
predicted_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
predicted_tree.bind("<<TreeviewSelect>>", on_predicted_tree_select)

# --- Load JSON Data and Build the Main Tree ---
medication_data = load_medication_data('medication.json')

# Extract unique symptoms from the JSON data and update the combobox values.
common_symptoms = extract_symptoms(medication_data)
symptom_dropdown_new["values"] = common_symptoms
print("Extracted Common Symptoms:", common_symptoms)

medication_details = {}

for category in sorted(medication_data.keys()):
    cat_data = medication_data[category]
    # Insert only the category node (do not add child nodes for description or symptoms)
    category_id = tree.insert("", tk.END, text=category, open=False)
    for treatment_type, meds in cat_data.items():
        # Skip the CategoryDescription and Symptoms fields.
        if treatment_type in ["CategoryDescription", "Symptoms"]:
            continue
        treatment_id = tree.insert(category_id, tk.END, text=treatment_type, open=False)
        if isinstance(meds, dict):
            for med_name, details in meds.items():
                med_id = tree.insert(treatment_id, tk.END, text=med_name)
                details_copy = details.copy()
                details_copy["Illness"] = category
                medication_details[med_id] = details_copy

# Optionally, add an aggregated node for all conventional medicines.
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

# Gather all nodes for the auto-suggestion feature.
all_nodes = []


def gather_nodes(parent=""):
    for item in tree.get_children(parent):
        all_nodes.append((item, tree.item(item, "text")))
        gather_nodes(item)


gather_nodes()

suggestion_mapping = {}

root.mainloop()

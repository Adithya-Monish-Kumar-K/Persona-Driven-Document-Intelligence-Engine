import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch  # <<< THIS LINE IS THE FIX
from torch.utils.data import DataLoader
import os

# --- IMPORT THE ROUND 1A LOGIC ---
try:
    from process_pdfs import run_pipeline as run_1a_pipeline
    from process_pdfs import load_models as load_1a_models
except ImportError:
    print("❌ Error: Could not import 'process_pdfs.py'. Make sure your Round 1A script is in the same directory.")
    exit()

print("--- Automated Multi-Collection Training for Round 1B Semantic Model ---")

# --- Configuration ---
TRAINING_ROOT_DIR = Path("training_collections/")
MODEL_SAVE_PATH = 'models/sentence_transformer_finetuned'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- Check for GPU and set the device ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_finetuning_dataset(models_1a):
    """
    Creates a dataset for fine-tuning by iterating through all persona collections.
    """
    master_train_examples = []
    
    if not TRAINING_ROOT_DIR.exists():
        print(f"❌ Error: The training directory '{TRAINING_ROOT_DIR}' was not found.")
        return []
        
    collection_paths = [d for d in TRAINING_ROOT_DIR.iterdir() if d.is_dir()]

    if not collection_paths:
        print(f"❌ No training collections found in '{TRAINING_ROOT_DIR}'.")
        return []

    for collection_path in collection_paths:
        print(f"\n--- Processing Collection: {collection_path.name} ---")
        
        docs_dir = collection_path / "pdfs"
        pdf_files = list(docs_dir.glob("*.pdf"))
        persona_file = collection_path / "challenge1b_input.json"
        ground_truth_file = collection_path / "challenge1b_output.json"

        if not (docs_dir.exists() and pdf_files and persona_file.exists() and ground_truth_file.exists()):
            print(f"⚠️ Warning: Skipping collection '{collection_path.name}' due to missing files/folders.")
            continue

        print(f"Step 1: Extracting outlines from {len(pdf_files)} PDFs...")
        all_sections = []
        for pdf_file in pdf_files:
            outline_json = run_1a_pipeline(pdf_file, *models_1a)
            if outline_json.get("title"):
                all_sections.append(outline_json["title"])
            for item in outline_json.get("outline", []):
                all_sections.append(item['text'])
        
        if not all_sections:
            print("No sections extracted. Skipping.")
            continue

        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        with open(persona_file, 'r', encoding='utf-8') as f:
            persona_data = json.load(f)

        relevant_sections = {item['section_title'].strip() for item in ground_truth.get('extracted_sections', [])}
        query = f"{persona_data['persona']}: {persona_data['job_to_be_done']}"

        collection_examples = 0
        for section_text in all_sections:
            score = 1.0 if section_text.strip() in relevant_sections else 0.0
            master_train_examples.append(InputExample(texts=[query, section_text], label=score))
            collection_examples += 1
        
        print(f"Generated {collection_examples} training examples.")
    
    return master_train_examples

# --- Main Training Logic ---
if __name__ == '__main__':
    print(f"--- Using device: {DEVICE.upper()} ---")
    print("Loading Round 1A models to be used for feature extraction...")
    models_1a = load_1a_models()

    training_data = create_finetuning_dataset(models_1a)

    if not training_data:
        print("❌ Could not generate any training data. Exiting.")
        exit()

    print(f"\n--- Total training examples from all collections: {len(training_data)} ---")
    print("Step 2: Fine-tuning the Sentence Transformer model on the combined dataset...")

    model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Fine-tuned Round 1B model saved to {MODEL_SAVE_PATH}")
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import mysql.connector
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  

# Database configuration
db_config = {
 'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'club-sync'
}

# Initialize model
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def get_db_connection():
    return mysql.connector.connect(**db_config)

def generate_task_steps(prompt):
    prompt = prompt.strip()
    if not prompt:
        return []
    
    input_text = f"""Convert this project description into clear numbered steps:

    Project: {prompt}

    Steps:
    1."""
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    steps_match = re.search(r"Steps:\s*(.*)", full_text, re.DOTALL)
    
    if not steps_match:
        return []
    
    steps_text = steps_match.group(1)
    steps = []
    
    for line in steps_text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            step = re.sub(r"^\d+\.\s*", "", line)
            if step:
                steps.append(step)
    
    return steps[:10]

def insert_tasks_to_db(project_id, assignee_id, tasks):
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        for task in tasks:
            insert_query = """
            INSERT INTO projet_tache 
            (completed_date, date_created, description, due_date, label, 
             last_updated, priorite, progress, status, titre, assignee_id, projet_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),  # completed_date
                current_time,             # date_created
                task,                     # description
                (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),                      # due_date
                "Generated Task",          # label
                current_time,             # last_updated
                "Medium",                 # priorite
                0,                        # progress
                "todo",            # status
                task[:50],                # titre (truncated to 50 chars)
                assignee_id,
                project_id
            ))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

@app.route('/generate-tasks', methods=['POST'])
def handle_request():
    try:
        data = request.get_json()
        if not data or 'description' not in data or 'projet_id' not in data or 'assignee_id' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        project_desc = data['description']
        project_id = data['projet_id']
        assignee_id = data['assignee_id']
        
        steps = generate_task_steps(project_desc)
        
        if not steps:
            return jsonify({"error": "Could not generate tasks"}), 400
        
        # Insert tasks into database
        success = insert_tasks_to_db(project_id, assignee_id, steps)
        
        if not success:
            return jsonify({"error": "Failed to save tasks to database"}), 500
        
        return jsonify({
            "project_id": project_id,
            "assignee_id": assignee_id,
            "generated_tasks": steps,
            "count": len(steps),
            "message": "Tasks successfully generated and saved"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5200)
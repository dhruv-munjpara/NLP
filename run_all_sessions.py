"""
Master Test Runner for all 5 NLP Assignment Sessions
Course: Data Analytics & Data Science (DA & DS)
Student: Dhruv Munjpara
Institute: TOPS Technologies
"""

import os
import sys
import subprocess

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SESSIONS = [
    ("Session 1: Introduction to NLP & Use Cases", "Session-01_Introduction_to_NLP_and_Use_Cases/session_01_tasks.py"),
    ("Session 2: Text Cleaning & Preprocessing", "Session-02_Text_Cleaning_and_Preprocessing/session_02_tasks.py"),
    ("Session 3: Feature Engineering for Text", "Session-03_Feature_Engineering_for_Text/session_03_tasks.py"),
    ("Session 4: Text Classification Models", "Session-04_Text_Classification_Models/session_04_tasks.py"),
    ("Session 5: Deep Learning for NLP", "Session-05_Deep_Learning_for_NLP/session_05_tasks.py")
]

def run_session(name, script_rel_path):
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print(f"Script : {script_rel_path}")
    print("=" * 80 + "\n")
    
    script_full_path = os.path.join(os.path.dirname(__file__), script_rel_path)
    if not os.path.exists(script_full_path):
        print(f"Error: Script not found at {script_full_path}")
        return False
        
    result = subprocess.run([sys.executable, script_full_path], capture_output=False)
    return result.returncode == 0

def main():
    print("#" * 80)
    print("TOPS TECHNOLOGIES — NATURAL LANGUAGE PROCESSING (NLP) ALL 5 SESSIONS RUNNER")
    print("Student: Dhruv Munjpara | Repository: https://github.com/dhruv-munjpara/NLP.git")
    print("#" * 80)
    
    success_count = 0
    for name, script_path in SESSIONS:
        success = run_session(name, script_path)
        if success:
            success_count += 1
            
    print("\n" + "#" * 80)
    print(f"SUMMARY: {success_count}/{len(SESSIONS)} Sessions Completed Successfully!")
    print("#" * 80)

if __name__ == "__main__":
    main()


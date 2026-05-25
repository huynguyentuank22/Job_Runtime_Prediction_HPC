import os
import json
import re

dirs_to_process = [
    '/network-volume/MSB_GRPO/Job_Runtime_Prediction_HPC/ANL',
    '/network-volume/MSB_GRPO/Job_Runtime_Prediction_HPC/HCMUT'
]

for d in dirs_to_process:
    for filename in os.listdir(d):
        if filename.endswith('.ipynb'):
            filepath = os.path.join(d, filename)
            try:
                with open(filepath, 'r') as f:
                    nb = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
            
            modified = False
            for cell in nb.get('cells', []):
                if cell['cell_type'] == 'code':
                    source = cell['source']
                    for i, line in enumerate(source):
                        # Replace meta model
                        if '.meta_model_name =' in line and ('gradientboosting' in line or 'randomforest' in line):
                            new_line = re.sub(r'(\w+)\.meta_model_name\s*=\s*[\'"](gradientboosting|randomforest)[\'"]', r"\1.meta_model_name = 'xgboost'", line)
                            if new_line != line:
                                source[i] = new_line
                                modified = True
                        
                        # Replace single-line model_combinations
                        if 'model_combinations = [[' in line:
                            new_line = re.sub(r"model_combinations\s*=\s*\[.*\]", "model_combinations = [['xgboost', 'lightgbm', 'catboost'], ['lightgbm', 'xgboost', 'catboost'], ['catboost', 'xgboost', 'lightgbm']]", line)
                            if new_line != line:
                                source[i] = new_line
                                modified = True

                        # Replace multi-line combinations
                        if "['extratrees', 'randomforest', 'xgboost']" in line:
                            source[i] = line.replace("['extratrees', 'randomforest', 'xgboost']", "['xgboost', 'lightgbm', 'catboost']")
                            modified = True
                        if "['randomforest', 'mlp', 'gradientboosting']" in line:
                            source[i] = line.replace("['randomforest', 'mlp', 'gradientboosting']", "['lightgbm', 'xgboost', 'catboost']")
                            modified = True
                        if "['lasso', 'xgboost', 'extratrees']" in line:
                            source[i] = line.replace("['lasso', 'xgboost', 'extratrees']", "['catboost', 'xgboost', 'lightgbm']")
                            modified = True
                            
            if modified:
                print(f"Updated {filepath}")
                with open(filepath, 'w') as f:
                    json.dump(nb, f, indent=1)

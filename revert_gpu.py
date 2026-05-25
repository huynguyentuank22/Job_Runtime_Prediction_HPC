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
                        # Revert meta model back to gradientboosting
                        if '.meta_model_name =' in line and "'xgboost'" in line:
                            new_line = re.sub(r'(\w+)\.meta_model_name\s*=\s*[\'"]xgboost[\'"]', r"\1.meta_model_name = 'gradientboosting'", line)
                            if new_line != line:
                                source[i] = new_line
                                modified = True
                        
                        # Revert single-line model_combinations
                        if 'model_combinations = [[' in line and "'lightgbm'" in line and "'catboost'" in line:
                            new_line = re.sub(r"model_combinations\s*=\s*\[.*\]", "model_combinations = [['extratrees', 'randomforest', 'xgboost'], ['randomforest', 'mlp', 'gradientboosting'], ['lasso', 'xgboost', 'extratrees']]", line)
                            if new_line != line:
                                source[i] = new_line
                                modified = True

                        # Revert multi-line combinations
                        if "['xgboost', 'lightgbm', 'catboost']" in line:
                            source[i] = line.replace("['xgboost', 'lightgbm', 'catboost']", "['extratrees', 'randomforest', 'xgboost']")
                            modified = True
                        if "['lightgbm', 'xgboost', 'catboost']" in line:
                            source[i] = line.replace("['lightgbm', 'xgboost', 'catboost']", "['randomforest', 'mlp', 'gradientboosting']")
                            modified = True
                        if "['catboost', 'xgboost', 'lightgbm']" in line:
                            source[i] = line.replace("['catboost', 'xgboost', 'lightgbm']", "['lasso', 'xgboost', 'extratrees']")
                            modified = True
                            
            if modified:
                print(f"Reverted {filepath}")
                with open(filepath, 'w') as f:
                    json.dump(nb, f, indent=1)

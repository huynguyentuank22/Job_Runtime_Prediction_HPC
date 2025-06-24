import gzip
import os
import shutil
import pandas as pd

def extract_swf_gz(input_file, output_file=None):
    try:
        # Validate input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_file.split('\\')[1][:-3]  # Remove .gz
                
        # Extract the file
        with gzip.open(input_file, 'rb') as f_in:
            with open('output_swf\\' + output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        print(f"Successfully extracted: {output_file}")
        return output_file
        
    except gzip.BadGzipFile:
        raise IOError("Invalid gzip file format")
    except Exception as e:
        raise IOError(f"Error extracting file: {str(e)}")

def parse_swf_file(file_path):
    # Define column names based on standard SWF format
    columns = [
        'job_id', 'submit_time', 'wait_time', 'run_time', 'num_allocated_processors',
        'avg_cpu_time_used', 'used_memory', 'requested_processors', 'requested_time',
        'requested_memory', 'status', 'user_id', 'group_id', 'executable_id',
        'queue_id', 'partition_id', 'preceding_job_id', 'think_time'
    ]
    
    # Read data while skipping comment lines
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments (lines starting with semicolon)
            if not line.strip().startswith(';'):
                values = []
                for x in line.strip().split():
                    if x.strip() != '':
                        if ',' in x.strip():
                            values.append(x)
                        else:
                            values.append(float(x))
                    else:
                        values.append(-1)
                data.append(values)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('output_csv\\' + file_path.split('\\')[1] + '.csv', index=False)
    print(f"Successfully parsed: {file_path}")

# Example usage:
if __name__ == "__main__":
    # folder = os.listdir('raw_dataset')
    # folder = folder[26:]
    # print(folder)
    path = r'output_swf\HCMUT-SuperNodeXP-2017-1.0.swf'
    parse_swf_file(path)

    # for file in folder:
    #     extracted_file_swf = extract_swf_gz('raw_dataset\\' + file)
    #     parse_swf_file('output_swf\\' + extracted_file_swf)


        
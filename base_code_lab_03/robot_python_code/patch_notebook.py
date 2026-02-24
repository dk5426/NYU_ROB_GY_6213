import json
import re

file_path = '/Users/dhawalkabra/Documents/GitHub/NYU_ROB_GY_6213/base_code_lab_03/robot_python_code/ekf_simulation.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def g_function' in line for line in source):
            # We found the cell. Check if it's currently using the Y=Forward logic, and if so, revert it to X=Forward
            new_source = []
            for line in source:
                # If they are currently using Y=Forward (which I gave them earlier), revert it to X=Forward exactly like it was before
                line = re.sub(r'x_new\s*=\s*x\s*\+\s*s\s*\*\s*math\.sin\(th_mid\)', r'x_new   = x  + s * math.cos(th_mid)', line)
                line = re.sub(r'y_new\s*=\s*y\s*\+\s*s\s*\*\s*math\.cos\(th_mid\)', r'y_new   = y  + s * math.sin(th_mid)', line)
                
                # Matrix G_x
                if '[1, 0,  s * math.cos(th)]' in line:
                    line = line.replace('[1, 0,  s * math.cos(th)]', '[1, 0, -s * math.sin(th)]')
                if '[0, 1, -s * math.sin(th)]' in line:
                    line = line.replace('[0, 1, -s * math.sin(th)]', '[0, 1,  s * math.cos(th)]')
                
                # Matrix G_u
                if '[math.sin(th) * ds_dec,  0.0         ]' in line:
                    line = line.replace('[math.sin(th) * ds_dec,  0.0         ]', '[math.cos(th) * ds_dec,  0.0         ]')
                if '[math.cos(th) * ds_dec,  0.0         ]' in line:
                    line = line.replace('[math.cos(th) * ds_dec,  0.0         ]', '[math.sin(th) * ds_dec,  0.0         ]')
                    
                new_source.append(line)
            cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")

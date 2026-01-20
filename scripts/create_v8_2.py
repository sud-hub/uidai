import json
import shutil

# Copy v8.1 to v8.2
source = r"d:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.1.ipynb"
dest = r"d:\Sudarshan Khot\Coding\UIDAI\Child_MBU_Predictive_Dropout_Model_v8.2.ipynb"

print(f"Copying {source}")
print(f"To {dest}")

shutil.copy(source, dest)

print("\nFile copied successfully!")
print(f"Created: {dest}")

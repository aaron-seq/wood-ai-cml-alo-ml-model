import sys
import os
import app.main
import inspect

print(f"app.main location: {app.main.__file__}")
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

# Check the score_cml_data function source
print("\nSource of score_cml_data:")
print(inspect.getsource(app.main.score_cml_data))

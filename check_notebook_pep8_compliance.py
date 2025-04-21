import subprocess
import sys
import os

def check_pep8_compliance(path_to_notebook):
    if not os.path.isfile(path_to_notebook):
        print(f"File not found: {path_to_notebook}")
        return

    print(f"PEP8 check for: {path_to_notebook}")
    result = subprocess.run(['nbqa', 'flake8', path_to_notebook, '--max-line-length=120'], capture_output=True, text=True)

    if result.returncode == 0:
        print('PEP8 compliance check passed!')
    else:
        print('PEP8 issues found:')
        print(result.stdout)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python check_notebook_pep8_compliance.py <notebook_file.ipynb>')
        sys.exit(1)

    notebook_file = sys.argv[1]
    check_pep8_compliance(notebook_file)
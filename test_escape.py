"""
Test: what file content produces valid browser JS for lines.join('\n')
"""
import subprocess
import tempfile
import os

file_contents = [
    "lines.join('\\n')",      # 2 backslashes in file
    "lines.join('\\\\n')",    # 4 backslashes in file
    "lines.join('\\\\\\n')",  # 6 backslashes in file
    "lines.join('\\\\\\\\n')", # 8 backslashes in file
]

for fc in file_contents:
    # Simulate: this content is inside a Python triple-quoted string in agent.py
    py_code = f'''
result = """const blob = new Blob([{fc}]);"""
print(result)
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(py_code)
        tmp = f.name

    out = subprocess.run(
        ['D:/program/Anaconda/envs/assist/python.exe', tmp],
        capture_output=True, text=True, encoding='utf-8'
    )
    os.unlink(tmp)
    browser_output = out.stdout.strip()
    print(f'File content:  {repr(fc)}')
    print(f'Browser gets:  {repr(browser_output)}')

    # Check: does browser get literal newline inside JS string (syntax error)?
    # Valid JS: lines.join('\n')  -> \n is escape sequence, becomes newline char
    # Invalid: lines.join('actual newline') -> syntax error
    has_literal_newline = '\n' in browser_output and r'\n' not in browser_output.split('join(')[1][:10]
    print(f'Has syntax error: {has_literal_newline}')
    print()

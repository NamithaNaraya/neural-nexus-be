import py_compile
import sys

files = [
    'app/combined_chat/fastrp_service.py',
    'app/combined_chat/embedding_service.py',
    'app/combined_chat/rag_service.py',
    'app/combined_chat/gds_service.py',
]

all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'  OK: {f}')
    except py_compile.PyCompileError as e:
        print(f'  FAIL: {f} - {e}')
        all_ok = False

print('\nAll OK' if all_ok else '\nFAILED')
sys.exit(0 if all_ok else 1)

import py_compile
import sys

files = [
    'app/combined_chat/fastrp_service.py',
    'app/combined_chat/embedding_service.py',
    'app/combined_chat/rag_service.py',
]

all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'  ✅ {f}: SYNTAX OK')
    except py_compile.PyCompileError as e:
        print(f'  ❌ {f}: SYNTAX ERROR - {e}')
        all_ok = False

if all_ok:
    print('\n✅ All files passed syntax check!')
else:
    print('\n❌ Some files have syntax errors')
    sys.exit(1)

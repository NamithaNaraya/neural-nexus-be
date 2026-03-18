import py_compile
import sys

try:
    py_compile.compile('app/combined_chat/embedding_service.py', doraise=True)
    print('embedding_service.py: SYNTAX OK')
except py_compile.PyCompileError as e:
    print(f'embedding_service.py: SYNTAX ERROR - {e}')
    sys.exit(1)

try:
    py_compile.compile('app/combined_chat/rag_service.py', doraise=True)
    print('rag_service.py: SYNTAX OK')
except py_compile.PyCompileError as e:
    print(f'rag_service.py: SYNTAX ERROR - {e}')
    sys.exit(1)

print('\nAll files passed syntax check!')

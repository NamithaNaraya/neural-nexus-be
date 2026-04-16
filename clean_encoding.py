import os

file_path = r'e:\Neural-Nexus-V1\frontend-react-v2\src\pages\chat\ChatPage.jsx'

# Read raw bytes
with open(file_path, 'rb') as f:
    raw = f.read()

print(f"Original size: {len(raw)} bytes")
print(f"Null bytes: {raw.count(b'\\x00')}")
print(f"BOM present: {raw[:3] == bytes([0xEF, 0xBB, 0xBF])}")

# Strip BOM if present
if raw[:3] == bytes([0xEF, 0xBB, 0xBF]):
    raw = raw[3:]
    print("Stripped BOM")

# Strip null bytes
cleaned = raw.replace(b'\\x00', b'')
if len(cleaned) != len(raw):
    print(f"Stripped {len(raw) - len(cleaned)} null bytes")

# Decode
content = cleaned.decode('utf-8')

# Normalize line endings to \r\n (Windows)
content = content.replace('\\r\\n', '\\n').replace('\\r', '\\n').replace('\\n', '\\r\\n')

# Write back clean
with open(file_path, 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print(f"Clean size: {len(content)} chars")
print("File cleaned successfully!")

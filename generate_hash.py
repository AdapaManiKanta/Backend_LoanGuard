"""
generate_hash.py — Generate bcrypt hash for seeding users table.
Usage: python generate_hash.py <password>
"""
import sys
import bcrypt

if len(sys.argv) < 2:
    print("Usage: python generate_hash.py <password>")
    sys.exit(1)

password = sys.argv[1].encode("utf-8")
hashed = bcrypt.hashpw(password, bcrypt.gensalt(12))
print(hashed.decode("utf-8"))

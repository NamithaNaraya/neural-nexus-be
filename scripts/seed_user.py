"""
Seed Test User Script

Creates a test user in the database for development/testing.
Run with: python scripts/seed_user.py
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.connections import get_postgres_session, init_postgres, close_postgres
from app.core.security import hash_password


async def create_test_user():
    """Create a test user for development."""
    
    # Test user credentials
    email = "admin@neuralnexus.ai"
    password = "Neural@2026"  # Plain text - will be hashed
    role = "admin"
    
    password_hash = hash_password(password)
    
    async with get_postgres_session() as session:
        # Check if user exists
        result = await session.execute(
            text("SELECT id FROM neural_nexus.users WHERE email = :email"),
            {"email": email}
        )
        existing = result.fetchone()
        
        if existing:
            print(f"✅ User already exists: {email}")
            return
        
        # Create user
        await session.execute(
            text("""
                INSERT INTO neural_nexus.users (email, password_hash, role)
                VALUES (:email, :password_hash, :role)
            """),
            {
                "email": email,
                "password_hash": password_hash,
                "role": role,
            }
        )
        await session.commit()
        
        print("=" * 50)
        print("✅ Test user created successfully!")
        print("=" * 50)
        print(f"📧 Email:    {email}")
        print(f"🔑 Password: {password}")
        print(f"👤 Role:     {role}")
        print("=" * 50)


async def create_regular_user():
    """Create a regular (non-admin) test user."""
    
    email = "user@neuralnexus.ai"
    password = "User@2026"
    role = "user"
    
    password_hash = hash_password(password)
    
    async with get_postgres_session() as session:
        result = await session.execute(
            text("SELECT id FROM neural_nexus.users WHERE email = :email"),
            {"email": email}
        )
        existing = result.fetchone()
        
        if existing:
            print(f"✅ User already exists: {email}")
            return
        
        await session.execute(
            text("""
                INSERT INTO neural_nexus.users (email, password_hash, role)
                VALUES (:email, :password_hash, :role)
            """),
            {
                "email": email,
                "password_hash": password_hash,
                "role": role,
            }
        )
        await session.commit()
        
        print(f"📧 Email:    {email}")
        print(f"🔑 Password: {password}")
        print(f"👤 Role:     {role}")
        print("=" * 50)


async def main():
    print("🔌 Initializing database connection...")
    await init_postgres()
    
    try:
        print("\n🌱 Seeding test users...\n")
        await create_test_user()
        await create_regular_user()
        print("\n✨ Done! You can now login with these credentials.\n")
    finally:
        print("🔌 Closing database connection...")
        await close_postgres()


if __name__ == "__main__":
    asyncio.run(main())

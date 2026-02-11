
import asyncio
from sqlalchemy import text
from app.db.connections import get_postgres_session, init_postgres, close_postgres

async def add_progress_column():
    print("Initializing PostgreSQL connection...")
    try:
        await init_postgres()
    except Exception as e:
        print(f"Failed to initialize PostgreSQL: {e}")
        return

    print("Adding 'progress' column to neural_nexus.files table...")
    try:
        session = get_postgres_session()
        async with session:
            # Check if column exists
            result = await session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'neural_nexus' 
                  AND table_name = 'files' 
                  AND column_name = 'progress'
            """))
            if result.fetchone():
                print("Column 'progress' already exists.")
            else:
                # Add column
                await session.execute(text("ALTER TABLE neural_nexus.files ADD COLUMN progress INTEGER DEFAULT 0"))
                await session.commit()
                print("Successfully added 'progress' column.")
    except Exception as e:
        print(f"Error adding column: {e}")
    finally:
        await close_postgres()

if __name__ == "__main__":
    asyncio.run(add_progress_column())

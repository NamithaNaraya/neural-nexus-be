import asyncio
from app.db.connections import init_postgres, get_postgres_session, close_postgres
from sqlalchemy import text

async def list_folders():
    await init_postgres()
    async with get_postgres_session() as session:
        result = await session.execute(
            text("SELECT id, name FROM neural_nexus.folders")
        )
        rows = result.fetchall()
        print("Folders in DB:")
        for row in rows:
            print(f" - ID: {row[0]} | Name: '{row[1]}'")
    await close_postgres()

if __name__ == "__main__":
    asyncio.run(list_folders())

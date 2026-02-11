import asyncio
from app.db.connections import init_postgres, get_postgres_session, close_postgres
from sqlalchemy import text

async def list_files():
    await init_postgres()
    folder_id = "77808f19-7db1-49fc-ab01-6565683d4c45"
    async with get_postgres_session() as session:
        result = await session.execute(
            text("SELECT id, filename, status FROM neural_nexus.files WHERE folder_id = :f_id"),
            {"f_id": folder_id}
        )
        rows = result.fetchall()
        print(f"Files in folder {folder_id}:")
        for row in rows:
            print(f" - ID: {row[0]} | Name: '{row[1]}' | Status: {row[2]}")
    await close_postgres()

if __name__ == "__main__":
    asyncio.run(list_files())

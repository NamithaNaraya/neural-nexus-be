import asyncio
from app.db.connections import init_neo4j, get_neo4j_driver, close_neo4j

async def inspect_nodes():
    await init_neo4j()
    driver = get_neo4j_driver()
    async with driver.session() as session:
        print("--- Inspecting Recent Nodes (last 50) ---")
        result = await session.run("""
            MATCH (n)
            RETURN labels(n) as labels, 
                   n.name as name, 
                   n.file_id as file_id, 
                   n.folder_id as folder_id,
                   n.id as id
            LIMIT 50
        """)
        async for record in result:
            print(f"Node: {record['name']} | Labels: {record['labels']} | File: {record['file_id']} | Folder: {record['folder_id']} | UUID: {record['id']}")
            
        print("\n--- Orphan Summary ---")
        res = await session.run("MATCH (n) WHERE n.file_id IS NULL OR n.folder_id IS NULL RETURN count(n) as orphans")
        count = (await res.single())["orphans"]
        print(f"Total partial orphans (missing file or folder): {count}")

        res = await session.run("MATCH (n) WHERE NOT n:Entity RETURN count(n) as non_entities")
        count = (await res.single())["non_entities"]
        print(f"Nodes missing :Entity label: {count}")

    await close_neo4j()

if __name__ == '__main__':
    asyncio.run(inspect_nodes())

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# Load from environment
MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]

def get_database():
    return db

async def get_all_jobs():
    """Helper to fetch all jobs from the collection."""
    cursor = db["jobs"].find({})
    return await cursor.to_list(length=1000)

async def get_all_occupations():
    """Helper to fetch all occupations from the collection."""
    cursor = db["occupations"].find({})
    return await cursor.to_list(length=1000)

async def close_mongo_connection():
    client.close()

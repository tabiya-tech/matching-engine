
import os
import json
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
    """Helper to fetch all occupations from the collection, with fallback to local JSON if DB fails."""
    try:
        cursor = db["occupations"].find({})
        return await cursor.to_list(length=1000)
    except Exception as e:
        # Fallback to local JSON file
        json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "combined_occupation_database_with_wa.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                occupations = json.load(f)
            return occupations
        except Exception as file_e:
            raise RuntimeError(f"Failed to fetch occupations from DB and fallback JSON: {e}, {file_e}")

async def close_mongo_connection():
    client.close()

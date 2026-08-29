from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from .config import settings


client = MongoClient(
    settings.MONGODB_URI,
    serverSelectionTimeoutMS=5000,
)

database: Database = client[
    settings.DATABASE_NAME
]



# COLLECTIONS
analyses_collection: Collection = (
    database["analyses"]
)

predictions_collection: Collection = (
    database["predictions"]
)


# DATABASE HEALTH
def check_database_connection() -> bool:
    """
    Check whether MongoDB is reachable.
    """

    try:

        client.admin.command("ping")

        return True

    except Exception:

        return False

# SHUTDOWN
def close_database_connection() -> None:
    """
    Close MongoDB connection.
    """

    client.close()
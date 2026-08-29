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

predictions_collection: Collection = (
    database["predictions"]
)


def check_database_connection() -> bool:

    try:

        client.admin.command("ping")

        return True

    except Exception as exc:

        print(
            f"MongoDB connection error: {exc}"
        )

        return False


def close_database_connection() -> None:

    client.close()
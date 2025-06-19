from pymongo import MongoClient
from urllib.parse import quote_plus

username = quote_plus("praveensrinivasan05")  # no special chars? safe
password = quote_plus("r!979088Q")  # special char "!" must be encoded
cluster = "cluster0.qqny5wz.mongodb.net"
database = "test"  # or your DB name

uri = f"mongodb+srv://{username}:{password}@{cluster}/{database}?retryWrites=true&w=majority&authSource=admin"

client = MongoClient(uri)
client.admin.command('ping')
print("✅ Connected to MongoDB Atlas")
def get_db():
    return client["test"]
import os
import django
import shutil

# 1️⃣ Set Django settings module (replace 'identiface' with your project name if different)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "identiface.settings")

# 2️⃣ Setup Django
django.setup()

# 3️⃣ Now import your models
from recognition.models import Person, FaceEmbedding, UnknownFace
from django.conf import settings

# 4️⃣ Delete all database entries
FaceEmbedding.objects.all().delete()
Person.objects.all().delete()
UnknownFace.objects.all().delete()

# 5️⃣ Delete face images from media folder
media_root = settings.MEDIA_ROOT
if os.path.exists(media_root):
    for item in os.listdir(media_root):
        item_path = os.path.join(media_root, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")

print("All face data and media files cleared!")

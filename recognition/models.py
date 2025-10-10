from django.db import models
import numpy as np
from django.conf import settings
import os

class Person(models.Model):
    """Model to store person information and their face embeddings."""
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class FaceEmbedding(models.Model):
    """Model to store face embeddings for each person."""
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='embeddings')
    embedding = models.BinaryField()  # Store serialized numpy array
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_embedding(self):
        """Deserialize the stored embedding back to numpy array."""
        return np.frombuffer(self.embedding, dtype=np.float32)
    
    @classmethod
    def create_from_embedding(cls, person, embedding):
        """Helper method to create a FaceEmbedding from a numpy array."""
        return cls.objects.create(
            person=person,
            embedding=embedding.tobytes()
        )
    
    class Meta:
        ordering = ['-created_at']

def get_or_create_person(name):
    """Helper function to get or create a person."""
    return Person.objects.get_or_create(name=name)[0]

def get_all_embeddings():
    """Retrieve all embeddings from the database."""
    embeddings_dict = {}
    for person in Person.objects.prefetch_related('embeddings').all():
        embeddings_dict[person.name] = [
            emb.get_embedding() for emb in person.embeddings.all()
        ]
    return embeddings_dict

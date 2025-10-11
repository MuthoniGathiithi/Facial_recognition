from django.db import models
import numpy as np
from django.conf import settings
import os
import uuid
from django.core.files.base import ContentFile
import cv2

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


class UnknownFace(models.Model):
    """Model to temporarily store unknown faces for later identification."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    embedding = models.BinaryField()  # Store serialized numpy array
    face_image = models.ImageField(upload_to='unknown_faces/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_embedding(self):
        """Deserialize the stored embedding back to numpy array."""
        return np.frombuffer(self.embedding, dtype=np.float32)
    
    @classmethod
    def create_from_face(cls, embedding, face_image_np):
        """Create an UnknownFace from a numpy array and face image."""
        # Convert numpy array to image
        _, buffer = cv2.imencode('.png', cv2.cvtColor(face_image_np, cv2.COLOR_RGB2BGR))
        face_image = ContentFile(buffer.tobytes(), 'unknown_face.png')
        
        return cls.objects.create(
            embedding=embedding.tobytes(),
            face_image=face_image
        )
    
    def enroll_as_person(self, name):
        """Enroll this unknown face as a new person."""
        person = get_or_create_person(name)
        FaceEmbedding.create_from_embedding(person, self.get_embedding())
        self.delete()  # Remove from unknown faces after enrollment
        return person

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

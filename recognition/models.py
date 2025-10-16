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
    pose = models.CharField(max_length=20, default='front', help_text='Pose: front, left, right, up, down')
    image_path = models.CharField(max_length=500, blank=True, help_text='Path to the captured image')
    quality_metrics = models.JSONField(default=dict, help_text='Quality assessment metrics')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_embedding(self):
        """Deserialize the stored embedding back to numpy array (512D for buffalo_l)."""
        embedding = np.frombuffer(self.embedding, dtype=np.float32)
        # Ensure consistent shape - 1D array (512D for buffalo_l model)
        if len(embedding.shape) > 1:
            embedding = embedding.reshape(-1)
        return embedding
    
    @classmethod
    def create_from_embedding(cls, person, embedding):
        """Helper method to create a FaceEmbedding from a numpy array.
        
        Args:
            person: The Person instance to associate with this embedding
            embedding: Numpy array of shape (1, 512) or (512,)
        """
        # Ensure the embedding is a 1D array before saving
        if len(embedding.shape) > 1:
            embedding = embedding.reshape(-1)
        
        # Debug print the shape before saving
        print(f"Saving embedding with shape: {embedding.shape}")
        
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
        return np.frombuffer(self.embedding, dtype=np.float32).reshape (1, -1)
    
    @classmethod
    def create_from_face(cls, embedding, face_image_np):
        """Create an UnknownFace from a numpy array and face image.
        
        Args:
            embedding: Numpy array of shape (1, 512) or (512,)
            face_image_np: Numpy array of the face image in RGB format
        """
        # Debug print the input embedding shape
        print(f"[DEBUG] create_from_face - Input embedding shape: {embedding.shape}")
        
        # Ensure the embedding is a 1D array before saving
        if len(embedding.shape) > 1:
            embedding = embedding.reshape(-1)
            
        print(f"[DEBUG] create_from_face - Reshaped embedding to: {embedding.shape}")
        
        try:
            # Convert numpy array to image
            _, buffer = cv2.imencode('.png', cv2.cvtColor(face_image_np, cv2.COLOR_RGB2BGR))
            face_image = ContentFile(buffer.tobytes(), 'unknown_face.png')
            
            # Save the unknown face to the database
            unknown_face = cls.objects.create(
                embedding=embedding.tobytes(),
                face_image=face_image
            )
            
            print(f"[DEBUG] Successfully created UnknownFace with ID: {unknown_face.id}")
            return unknown_face
            
        except Exception as e:
            print(f"[ERROR] Failed to create UnknownFace: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller
    
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
    """Retrieve all embeddings from database.
    
    Returns:
        dict: A dictionary mapping person names to lists of their face embeddings.
        Each embedding is a numpy array (512D for buffalo_l model).
    """
    embeddings_dict = {}
    
    # Load from database
    try:
        for person in Person.objects.prefetch_related('embeddings').all():
            person_embeddings = []
            for emb in person.embeddings.all():
                embedding = emb.get_embedding()
                # Ensure consistent shape - 1D array (512D for buffalo_l)
                if len(embedding.shape) > 1:
                    embedding = embedding.reshape(-1)
                person_embeddings.append(embedding)
            
            if person_embeddings:  # Only add if there are embeddings
                embeddings_dict[person.name] = person_embeddings
        
        print(f"📊 Loaded {len(embeddings_dict)} people from database")
        
    except Exception as e:
        print(f"⚠️ Database access failed: {str(e)}")
    
    # Summary
    total_people = len(embeddings_dict)
    print(f"✅ Total available: {total_people} people")
    for name, embs in embeddings_dict.items():
        print(f"  - {name}: {len(embs)} embeddings")
    
    return embeddings_dict

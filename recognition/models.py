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


def get_or_create_person(name):
    """Helper function to get or create a person."""
    return Person.objects.get_or_create(name=name)[0]

# Simple cache for embeddings to improve matching speed
_embeddings_cache = None
_cache_timestamp = None

def get_all_embeddings():
    """Retrieve all embeddings from database with caching for speed.
    
    Returns:
        dict: A dictionary mapping person names to lists of their face embeddings.
        Each embedding is a numpy array (512D for buffalo_l model).
    """
    global _embeddings_cache, _cache_timestamp
    import time
    
    # Use cache if it's less than 5 minutes old (much longer cache)
    current_time = time.time()
    if _embeddings_cache is not None and _cache_timestamp is not None:
        if current_time - _cache_timestamp < 300:  # 5 minute cache for speed
            print("⚡ Using cached embeddings for faster matching")
            return _embeddings_cache
    
    print("📂 Loading fresh embeddings from database...")
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
    
    # Update cache
    _embeddings_cache = embeddings_dict
    _cache_timestamp = current_time
    
    # Summary
    total_people = len(embeddings_dict)
    print(f"✅ Total available: {total_people} people")
    for name, embs in embeddings_dict.items():
        print(f"  - {name}: {len(embs)} embeddings")
    
    return embeddings_dict

def clear_embeddings_cache():
    """Clear the embeddings cache to force fresh loading on next request."""
    global _embeddings_cache, _cache_timestamp
    _embeddings_cache = None
    _cache_timestamp = None
    print("🗑️ Embeddings cache cleared")

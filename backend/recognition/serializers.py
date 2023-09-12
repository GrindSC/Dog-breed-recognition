from rest_framework import serializers
from .models import UserImages

class UserImagesSerializer(serializers.ModelSerializer):
    class Meta:
        model=UserImages
        fields='__all__'
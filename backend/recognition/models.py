from django.db import models

class UserImages(models.Model):
    image=models.ImageField()
    width=models.DecimalField(decimal_places=1, max_digits=20)
    height=models.DecimalField(decimal_places=1, max_digits=20)

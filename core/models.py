from django.db import models

# Create your models here.



class Documents(models.Model):
    file = models.FileField(upload_to='pdf')


    def __str__(self):
        return f"{self.file}"
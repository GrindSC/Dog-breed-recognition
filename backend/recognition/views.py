from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UserImages
from .serializers import UserImagesSerializer

# from django.shortcuts import render_to_response
from django import forms
class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Select a file',
        help_text='max. 42 megabytes'
    )

@api_view(['GET'])
def getUserImages(request):
    uploads = UserImages.objects.all()
    serializer=UserImagesSerializer(uploads, many=True)
    return Response(serializer.data)

# @api_view(['PUT'])
# def uploadUserImages(request):
#     # uploads = UserImages.objects.all()
#     serializer=UserImagesSerializer(data=request.data)
#     print(serializer.data)
#     return Response(serializer.data)

@api_view(['PUT'])
def uploadUserImages(request):
    context = {}
    form = DocumentForm(request.POST, request.FILES)
    context['form']= form
    # return render(request, "home.html", context)
    return Response(form)
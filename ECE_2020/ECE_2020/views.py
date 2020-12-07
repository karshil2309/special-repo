from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings


def a(request):
    return render(request,'index.html')

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'r.wav', {
            'uploaded_file_url': uploaded_file_url
        })

    return render(request, 'r.wav')
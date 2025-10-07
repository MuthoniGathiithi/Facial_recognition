from django.shortcuts import render

def enroll(request):
    message = ""  
    if request.method == 'POST':
        name = request.POST.get('name')
        photo = request.POST.get('photo')

      
        print("Name:", name)
        print("Photo (first 50 chars):", photo[:50])

        message = "Face enrolled successfully!"  

    return render(request, 'enroll.html', {'message': message})

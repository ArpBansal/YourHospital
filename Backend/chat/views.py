from django.shortcuts import render

# Create your views here.

def AIchatViews(request):
    return render(request, 'chat_frontend.html')

from django.shortcuts import render

# Create your views here.
def home(request):
    import requests
    import json
    api_request = requests.get("https://api.github.com/users?since=0")
    api = json.loads(api_request.content)
    return render(request,'home.html',{"api":api})

def user(request):
    if request.method == 'POST':
        user = request.POST['user']
        import requests
        import json
        user_request = requests.get("https://api.github.com/users/"+user)
        username = json.loads(user_request.content)
        return render(request,'user.html',{'user':user,'username':username})
    else:
        notfound = "请在搜索框输入查询的用户"
        return render(request,'user.html',{'notfound':notfound})
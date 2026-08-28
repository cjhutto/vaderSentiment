from django.urls import include, path

urlpatterns = [
    path("", include("analyzer.urls")),
]

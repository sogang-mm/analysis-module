from django.conf.urls import url
from django.conf.urls import include
from rest_framework.routers import DefaultRouter
from WebAnalyzer import views


router = DefaultRouter()

router.register(r'image', views.ImageViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
]

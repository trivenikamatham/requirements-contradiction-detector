from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    # Core app views
    path('', views.upload_view, name='upload'),
    path('sessions/', views.session_list_view, name='session_list'),
    path('sessions/<int:session_id>/report/', views.report_view, name='report'),
    path('sessions/<int:session_id>/delete/', views.delete_session_view, name='delete_session'),
    path('sample/load/', views.load_sample_view, name='load_sample'),
    path('export/<int:session_id>/<str:format>/', views.export_report_view, name='export_report'),

    # Authentication views
    path('auth/login/', views.login_view, name='login'),
    path('auth/register/', views.register_view, name='register'),
    path('auth/logout/', views.logout_view, name='logout'),
    path('auth/profile/', views.profile_view, name='profile'),
]

from django.urls import path

from . import views

urlpatterns = [
    path("", views.analyze, name="analyze"),
    path("api/analyze/", views.api_analyze, name="api_analyze"),
    path("api/analyze-url/", views.api_analyze_url, name="api_analyze_url"),
    path("api/analyze-document/", views.api_analyze_document, name="api_analyze_document"),
    path("api/analyze-batch/", views.api_analyze_batch, name="api_analyze_batch"),
    path("export/batch-csv/", views.export_batch_csv, name="export_batch_csv"),
    path("report/", views.view_report, name="view_report"),
    path("export/report-html/", views.export_report_html, name="export_report_html"),
    path("export/report-json/", views.export_report_json, name="export_report_json"),
]

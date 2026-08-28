from django.conf import settings


def analytics(request):
    return {
        "google_analytics_id": settings.GOOGLE_ANALYTICS_MEASUREMENT_ID,
    }

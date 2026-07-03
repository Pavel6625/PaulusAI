"""A tiny CORS middleware for the mini-app browser calls.

The Telegram Mini App runs on its own origin and calls this API with fetch(), so
it needs CORS headers. Only origins listed in DJANGO_CORS_ORIGINS are allowed;
an empty list disables cross-origin access. (Kept dependency-free; swap in
django-cors-headers if you need finer control.)
"""
from django.conf import settings


class SimpleCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        origin = request.headers.get("Origin")
        allowed = origin and origin in settings.CORS_ALLOWED_ORIGINS

        if request.method == "OPTIONS" and origin:
            from django.http import HttpResponse
            response = HttpResponse(status=204)
        else:
            response = self.get_response(request)

        if allowed:
            response["Access-Control-Allow-Origin"] = origin
            response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response["Vary"] = "Origin"
        return response

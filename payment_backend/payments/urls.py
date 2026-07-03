from django.urls import path

from . import views

urlpatterns = [
    # Public
    path("tonconnect-manifest.json", views.tonconnect_manifest,
         name="tonconnect-manifest"),

    # Mini app (Telegram initData auth)
    path("invoices", views.create_invoice_view, name="create-invoice"),
    path("invoices/<uuid:invoice_id>/confirm", views.confirm_invoice_view,
         name="confirm-invoice"),
    path("balance", views.balance_view, name="balance"),

    # Internal (PaulusAI, shared-token auth)
    path("internal/accounts/<int:telegram_id>/balance",
         views.internal_balance_view, name="internal-balance"),
    path("internal/debit", views.internal_debit_view, name="internal-debit"),
]

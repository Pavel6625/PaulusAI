from django.contrib import admin

from .models import Account, Invoice, PaymentTransaction


@admin.register(Account)
class AccountAdmin(admin.ModelAdmin):
    list_display = ("telegram_id", "balance_usd", "created_at")
    search_fields = ("telegram_id",)


@admin.register(Invoice)
class InvoiceAdmin(admin.ModelAdmin):
    list_display = ("memo", "account", "amount_usd", "amount_ton", "status",
                    "created_at")
    list_filter = ("status",)
    search_fields = ("memo",)


@admin.register(PaymentTransaction)
class PaymentTransactionAdmin(admin.ModelAdmin):
    list_display = ("tx_hash", "account", "currency", "amount_paid", "amount_usd",
                    "created_at")
    list_filter = ("currency",)
    search_fields = ("tx_hash", "from_address")

"""Diagnostic CLI for the payment validator — run it against the REAL TON API.

Use it once you have a funded wallet to confirm the on-chain parsing (especially
USD₮ jetton detection) matches reality before going live. It reads the same
config the validator uses (DP_TON_WALLET_ADDRESS, DP_USDT_JETTON_WALLET,
DP_TON_API_KEY, …) from the environment.

    python -m paulus.payments.inspect scan [--limit N] [--raw]
        Fetch recent inbound transactions to our wallet and show, for each, the
        raw message shape and how the classifier reads it. This is the fastest
        way to verify a real USD₮ payment shows up as currency=USDT.

    python -m paulus.payments.inspect tx <tx_hash> [--raw]
        Classify a single transaction by hash.

    python -m paulus.payments.inspect memo <memo>
        Run the real wallet scan for a memo (what Django's /find calls).

    python -m paulus.payments.inspect rate [--currency usd]
        Fetch the TON price used to quote invoices.

Nothing here writes anything; it only reads the chain.
"""
import argparse
import json
import sys

from .ton import PaymentVerificationError, TONClient, VerifiedPayment, _addr


def _client():
    return TONClient()


def _print_config(client):
    print("config:")
    print(f"  api_base           = {client.api_base}")
    print(f"  api_key            = {'set' if client.api_key else '(none)'}")
    print(f"  ton_wallet         = {client.wallet_address or '(unset!)'}")
    print(f"  usdt_jetton_wallet = {client.usdt_jetton_wallet or '(unset -> TON only)'}")
    print()


def _msg_summary(in_msg):
    """One-line-ish description of an inbound message's raw shape."""
    op = in_msg.get("decoded_op_name") or "(none)"
    src = _addr(in_msg.get("source")) or "(none)"
    dst = _addr(in_msg.get("destination")) or "(none)"
    value = in_msg.get("value")
    body = in_msg.get("decoded_body")
    body_keys = list(body.keys()) if isinstance(body, dict) else type(body).__name__
    return (f"    op={op}  value={value}\n"
            f"    source={src}\n"
            f"    dest={dst}\n"
            f"    decoded_body keys: {body_keys}")


def _verdict(client, tx):
    """Classifier verdict for a tx, as a printable string."""
    payment = client._classify(tx, tx.get("hash", ""))
    if payment is None:
        return "    [--] not recognised as a payment to us"
    return (f"    [OK] {payment.currency} {payment.amount:g}"
            f"  comment={payment.comment!r}  from={payment.from_address}")


def _show_tx(client, tx, raw=False):
    in_msg = tx.get("in_msg") or {}
    ok = not (tx.get("success") is False or tx.get("aborted") is True)
    print(f"  hash={tx.get('hash', '(?)')}  success={ok}  utime={tx.get('utime')}")
    print(_msg_summary(in_msg))
    print(_verdict(client, tx))
    if raw:
        print("    raw:")
        print(json.dumps(tx, indent=2, ensure_ascii=False)[:4000])
    print()


def cmd_scan(client, args):
    txs = client.fetch_wallet_transactions(args.limit)
    print(f"fetched {len(txs)} recent transaction(s) for {client.wallet_address}\n")
    recognised = 0
    for tx in txs:
        _show_tx(client, tx, raw=args.raw)
        if client._classify(tx, tx.get("hash", "")) is not None:
            recognised += 1
    print(f"summary: {recognised}/{len(txs)} classified as payments to us")
    return 0


def cmd_tx(client, args):
    tx = client.fetch_transaction(args.tx_hash)
    _show_tx(client, tx, raw=args.raw)
    return 0


def cmd_memo(client, args):
    try:
        payment: VerifiedPayment = client.find_payment(expected_comment=args.memo)
    except PaymentVerificationError as e:
        print(f"[--] no match: {e}")
        return 1
    print(f"[OK] found {payment.currency} {payment.amount:g} for memo "
          f"{args.memo!r}\n     tx={payment.tx_hash}  from={payment.from_address}")
    return 0


def cmd_rate(client, args):
    rate = client.get_rate(args.currency)
    print(f"1 TON = {rate} {args.currency.upper()}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="paulus.payments.inspect",
        description="Inspect real TON/USD₮ payments through the validator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="classify recent wallet transactions")
    p_scan.add_argument("--limit", type=int, default=20)
    p_scan.add_argument("--raw", action="store_true", help="dump raw JSON per tx")
    p_scan.set_defaults(func=cmd_scan)

    p_tx = sub.add_parser("tx", help="classify a single transaction by hash")
    p_tx.add_argument("tx_hash")
    p_tx.add_argument("--raw", action="store_true")
    p_tx.set_defaults(func=cmd_tx)

    p_memo = sub.add_parser("memo", help="run the real wallet scan for a memo")
    p_memo.add_argument("memo")
    p_memo.set_defaults(func=cmd_memo)

    p_rate = sub.add_parser("rate", help="fetch the TON price")
    p_rate.add_argument("--currency", default="usd")
    p_rate.set_defaults(func=cmd_rate)

    args = parser.parse_args(argv)
    client = _client()
    _print_config(client)
    try:
        return args.func(client, args)
    except PaymentVerificationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

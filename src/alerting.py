"""
alerting.py
-----------
Threshold-based alerting for the Rossmann Sales Forecasting model.

Design goals:
- Zero required external services: every alert is written to a single file,
  `src/log/alerts.log` (JSON lines, via monitoring.log_alert), which is also
  what the Monitoring Dashboard reads from — one plain-text file as the
  single source of truth, nothing else to keep in sync.
- Optional Slack notification if a `SLACK_WEBHOOK_URL` environment variable
  is set.
- Optional email notification if `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`,
  and `ALERT_EMAIL_TO` environment variables are set.
Both integrations fail silently (log-only) if not configured or if the
network call errors out, so alert *recording* never breaks the app.

Thresholds (override via environment variables):
- MAPE_ALERT_THRESHOLD   (default 0.20  -> alert if rolling MAPE > 20%)
- RMSE_ALERT_THRESHOLD   (default 1725  -> ~1.5x training RMSE of 1150.54)
- PSI_ALERT_THRESHOLD    (default 0.25  -> PSI >= 0.25 is "critical" drift)
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.mime.text import MIMEText

from monitoring import log_alert, compute_drift, compute_performance

MAPE_ALERT_THRESHOLD = float(os.environ.get("MAPE_ALERT_THRESHOLD", 0.20))
RMSE_ALERT_THRESHOLD = float(os.environ.get("RMSE_ALERT_THRESHOLD", 1725))
PSI_ALERT_THRESHOLD = float(os.environ.get("PSI_ALERT_THRESHOLD", 0.25))


def _notify_slack(message: str) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        data = json.dumps({"text": message}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:  # never let a notification failure break the app
        log_alert("warning", "alerting", f"Slack notification failed: {exc}")


def _notify_email(subject: str, message: str) -> None:
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("ALERT_EMAIL_TO")
    if not all([host, user, password, to_addr]):
        return
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587))) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
    except Exception as exc:
        log_alert("warning", "alerting", f"Email notification failed: {exc}")


def raise_alert(severity: str, source: str, message: str) -> None:
    """Central alert sink: alerts.log + (optional) Slack/email."""
    log_alert(severity, source, message)
    if severity in ("warning", "critical"):
        _notify_slack(f"[{severity.upper()}] {source}: {message}")
        _notify_email(f"[Rossmann Model Alert - {severity.upper()}] {source}", message)


def check_drift_alerts(window: int = 200) -> None:
    drift_df = compute_drift(window=window)
    if drift_df.empty:
        return
    critical = drift_df[drift_df["status"] == "critical"]
    warning = drift_df[drift_df["status"] == "warning"]
    for _, row in critical.iterrows():
        raise_alert("critical", "drift_monitor",
                    f"Feature '{row['feature']}' shows critical drift (PSI={row['psi']}).")
    for _, row in warning.iterrows():
        raise_alert("warning", "drift_monitor",
                    f"Feature '{row['feature']}' shows moderate drift (PSI={row['psi']}).")


def check_performance_alerts(window: int = 500) -> None:
    perf = compute_performance(window=window)
    if perf is None:
        return
    if perf["mape"] > MAPE_ALERT_THRESHOLD:
        raise_alert("critical", "performance_monitor",
                    f"Rolling MAPE ({perf['mape']:.2%}) exceeds threshold "
                    f"({MAPE_ALERT_THRESHOLD:.0%}) over last {perf['n_samples']} labeled predictions.")
    if perf["rmse"] > RMSE_ALERT_THRESHOLD:
        raise_alert("critical", "performance_monitor",
                    f"Rolling RMSE (€{perf['rmse']:,.0f}) exceeds threshold "
                    f"(€{RMSE_ALERT_THRESHOLD:,.0f}) over last {perf['n_samples']} labeled predictions.")


def run_all_checks() -> None:
    check_drift_alerts()
    check_performance_alerts()


if __name__ == "__main__":
    run_all_checks()
    print("Alert checks complete. See src/log/alerts.log and the monitoring dashboard.")

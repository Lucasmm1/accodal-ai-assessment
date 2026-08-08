SENSITIVE_FIELDS = {"name", "resident_name", "medical_history", "phone", "email", "address"}

def redact_sensitive_fields(data: dict) -> dict:
    return {
        key: "[REDACTED]" if key in SENSITIVE_FIELDS else value for key, value in data.items()
    }
from src.agents.schemas import IncidentClassification

class RegulatoryRouter:
    ROUTES = {
        "fall": "state_incident_notification",
        "medication_error": "medication_incident_notification",
        "abuse": "immediate_protective_services_notification",
    }

    def route(self, classification: IncidentClassification) -> str:
        return self.ROUTES.get(
            classification.incident_type,
            "manual_regulatory_review",
        )